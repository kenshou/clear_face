use crate::detector::{FaceBox, FaceDetector, FaceTracker};
use crate::overlay::AvatarOverlay;
use anyhow::{Context, Result, bail, ensure};
use image::RgbImage;
use rayon::prelude::*;
use std::collections::VecDeque;
use std::io::{Read, Write};
use std::path::Path;
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::thread;
use std::time::Instant;

#[derive(Debug, Clone)]
pub struct VideoMeta {
    pub width: u32,
    pub height: u32,
    pub fps_str: String,
    pub fps: f32,
    pub total_frames: Option<u64>,
    pub has_audio: bool,
}

struct BufferedFrame {
    image: RgbImage,
    faces: Vec<FaceBox>,
}

/// 将当前帧检测到的人脸回填到前序缓冲帧列表中（若对应位置尚未打码则补上相同位置的人脸框）
fn backfill_faces(
    history: &mut VecDeque<BufferedFrame>,
    current_faces: &[FaceBox],
    iou_threshold: f32,
) {
    if current_faces.is_empty() {
        return;
    }
    for past_frame in history.iter_mut() {
        for face in current_faces {
            let already_covered = past_frame
                .faces
                .iter()
                .any(|existing| existing.iou(face) > iou_threshold);
            if !already_covered {
                past_frame.faces.push(*face);
            }
        }
    }
}

fn parse_fps(fps_str: &str) -> f32 {
    if let Some((num, den)) = fps_str.split_once('/') {
        let n: f32 = num.parse().unwrap_or(30.0);
        let d: f32 = den.parse().unwrap_or(1.0);
        if d > 0.0 { n / d } else { 30.0 }
    } else {
        fps_str.parse().unwrap_or(30.0)
    }
}

pub fn probe_video(input: &Path) -> Result<VideoMeta> {
    let output = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate,nb_frames:stream_side_data=rotation",
            "-of",
            "default=noprint_wrappers=1",
        ])
        .arg(input)
        .output()
        .context("无法启动 ffprobe，请确认已安装 ffmpeg")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!("ffprobe 解析视频失败: {}", stderr.trim());
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut width: Option<u32> = None;
    let mut height: Option<u32> = None;
    let mut fps_str = String::from("30");
    let mut total_frames: Option<u64> = None;
    let mut rotation: i32 = 0;

    for line in stdout.lines() {
        if let Some((k, v)) = line.trim().split_once('=') {
            match k {
                "width" => width = v.parse().ok(),
                "height" => height = v.parse().ok(),
                "r_frame_rate" if !v.is_empty() && v != "0/0" => {
                    fps_str = v.to_string();
                }
                "nb_frames" => total_frames = v.parse().ok(),
                "rotation" => rotation = v.parse().unwrap_or(0),
                _ => {}
            }
        }
    }

    let mut w = width.context("未能从视频流中获取宽度 (width)")?;
    let mut h = height.context("未能从视频流中获取高度 (height)")?;

    // ffmpeg 解码默认自动应用旋转矩阵 (autorotate)，若旋转角为 ±90/±270 度则实际输出帧宽高互换
    if rotation.abs() % 180 == 90 {
        std::mem::swap(&mut w, &mut h);
    }

    ensure!(w > 0 && h > 0, "视频宽高无效: {}x{}", w, h);
    let fps = parse_fps(&fps_str);

    // 探测是否存在音频流
    let audio_out = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=codec_type",
            "-of",
            "csv=p=0",
        ])
        .arg(input)
        .output();
    let has_audio = audio_out
        .map(|o| o.status.success() && !String::from_utf8_lossy(&o.stdout).trim().is_empty())
        .unwrap_or(false);

    Ok(VideoMeta {
        width: w,
        height: h,
        fps_str,
        fps,
        total_frames,
        has_audio,
    })
}

pub fn process_video(
    input: &Path,
    output: &Path,
    detector: &FaceDetector,
    tracker: &mut FaceTracker,
    overlay: &AvatarOverlay,
    pre_seconds: f32,
    detect_interval: usize,
    sw_codec: bool,
    keep_audio: bool,
    voice_pitch: f32,
) -> Result<()> {
    let meta = probe_video(input)?;
    let pre_frames = if pre_seconds > 0.0 {
        (meta.fps * pre_seconds).ceil() as usize
    } else {
        0
    };
    let interval = detect_interval.max(1);
    let num_threads = rayon::current_num_threads();
    let batch_size = (num_threads * interval).clamp(8, 32);
    let use_hw = cfg!(target_os = "macos") && !sw_codec;
    let anonymize_audio = meta.has_audio && !keep_audio;

    println!(
        "视频信息: {}x{} @ {} fps ({:.2} fps){}, 并行线程: {}, 编解码: {}, 音频: {}, 检测间隔: 每 {} 帧, 向前补码: {:.2}s ({} 帧)",
        meta.width,
        meta.height,
        meta.fps_str,
        meta.fps,
        meta.total_frames
            .map(|n| format!(", 共 {} 帧", n))
            .unwrap_or_default(),
        num_threads,
        if use_hw {
            "VideoToolbox 硬件加速"
        } else {
            "CPU 软编解码"
        },
        if !meta.has_audio {
            "无音轨".to_string()
        } else if anonymize_audio {
            format!("人声频段隐私变声 (pitch={:.2})", voice_pitch)
        } else {
            "保留原声".to_string()
        },
        interval,
        pre_seconds.max(0.0),
        pre_frames
    );

    let mut dec_cmd = Command::new("ffmpeg");
    dec_cmd.args(["-v", "error"]);
    if use_hw {
        dec_cmd.args(["-hwaccel", "videotoolbox"]);
    }
    let mut decoder = dec_cmd
        .arg("-i")
        .arg(input)
        .args(["-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .context("启动 ffmpeg 解码子进程失败")?;

    let size_arg = format!("{}x{}", meta.width, meta.height);
    let mut enc_cmd = Command::new("ffmpeg");
    enc_cmd
        .args([
            "-y",
            "-v",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            &size_arg,
            "-r",
            &meta.fps_str,
            "-i",
            "pipe:0",
            "-i",
        ])
        .arg(input)
        .args(["-map", "0:v:0", "-map", "1:a?"]);

    if use_hw {
        enc_cmd.args([
            "-c:v",
            "h264_videotoolbox",
            "-q:v",
            "65",
            "-pix_fmt",
            "yuv420p",
        ]);
    } else {
        enc_cmd.args([
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
        ]);
    }

    if anonymize_audio {
        // 使用 16 阶 (-96dB/oct) 分频器切出 90Hz~4500Hz 人声频段进行共振峰偏移变声与微颤调制，保留低频/高频环境声不变
        let audio_filter = format!(
            "acrossover=split='90 4500':order=16th[low][voice][high];\
             [voice]rubberband=pitch={:.3}:formant=shifted:pitchq=quality,vibrato=f=5.5:d=0.12[v_mod];\
             [low][v_mod][high]amix=inputs=3:normalize=0",
            voice_pitch
        );
        enc_cmd.args(["-af", &audio_filter, "-c:a", "aac", "-b:a", "192k"]);
    } else {
        enc_cmd.args(["-c:a", "copy"]);
    }

    let mut encoder = enc_cmd
        .args(["-movflags", "+faststart"])
        .arg(output)
        .stdin(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .context("启动 ffmpeg 编码子进程失败")?;

    let mut dec_stdout = decoder.stdout.take().context("无法获取解码器 stdout")?;
    let mut enc_stdin = encoder.stdin.take().context("无法获取编码器 stdin")?;

    let frame_bytes = (meta.width as usize) * (meta.height as usize) * 3;
    let pool_capacity = batch_size * 3 + pre_frames + 4;

    // 内存缓冲区循环复用通道（Writer -> Reader），避免重复分配 6.2MB 1080p 帧内存
    let (free_tx, free_rx) = mpsc::sync_channel::<Vec<u8>>(pool_capacity);
    // 解码帧通道（Reader -> Main），容量设为 batch_size * 2 以保证解码器持续满速运行
    let (dec_tx, dec_rx) = mpsc::sync_channel::<Result<(u64, RgbImage)>>(batch_size * 2);
    // 待写入编码器帧通道（Main -> Writer），容量设为 batch_size * 2 以保证编码器持续满速运行
    let (enc_tx, enc_rx) = mpsc::sync_channel::<RgbImage>(batch_size * 2);

    let width = meta.width;
    let height = meta.height;

    // 1. 独立解码读取线程：持续从 ffmpeg 解码管道拉取帧，突破 macOS 64KB 管道阻塞瓶颈
    let reader_handle = thread::spawn(move || {
        let mut idx = 0u64;
        loop {
            let mut raw_buf = free_rx
                .try_recv()
                .unwrap_or_else(|_| vec![0u8; frame_bytes]);
            match read_exact_or_eof(&mut dec_stdout, &mut raw_buf) {
                Ok(true) => {
                    idx += 1;
                    match RgbImage::from_raw(width, height, raw_buf)
                        .context("构建帧图像缓冲区失败")
                    {
                        Ok(frame) => {
                            if dec_tx.send(Ok((idx, frame))).is_err() {
                                break;
                            }
                        }
                        Err(e) => {
                            let _ = dec_tx.send(Err(e));
                            break;
                        }
                    }
                }
                Ok(false) => break,
                Err(e) => {
                    let _ = dec_tx.send(Err(e));
                    break;
                }
            }
        }
    });

    // 2. 独立编码写入线程：持续向 ffmpeg 编码管道写入已贴图帧，并回收内存缓冲区
    let writer_handle = thread::spawn(move || -> Result<()> {
        while let Ok(frame) = enc_rx.recv() {
            enc_stdin
                .write_all(frame.as_raw())
                .context("向 ffmpeg 编码管道写入视频帧失败")?;
            let _ = free_tx.try_send(frame.into_raw());
        }
        drop(enc_stdin);
        Ok(())
    });

    // 3. 主协调流水线：批量接收解码帧 -> rayon 多核并行检测 -> 追踪与回填 -> rayon 多核并行贴图 -> 送入编码线程
    let mut history: VecDeque<BufferedFrame> = VecDeque::with_capacity(pre_frames + batch_size + 1);
    let mut frame_count = 0u64;
    let mut total_faces_detected = 0u64;
    let start_time = Instant::now();
    let mut eof = false;

    while !eof {
        let mut batch: Vec<(u64, RgbImage)> = Vec::with_capacity(batch_size);
        for _ in 0..batch_size {
            match dec_rx.recv() {
                Ok(Ok(item)) => {
                    frame_count = item.0;
                    batch.push(item);
                }
                Ok(Err(e)) => return Err(e),
                Err(_) => {
                    eof = true;
                    break;
                }
            }
        }

        if batch.is_empty() {
            break;
        }

        // 使用 rayon 多核并行对批次内需要检测的帧执行 ONNX 推理
        let batch_detections: Result<Vec<Option<Vec<FaceBox>>>> = batch
            .par_iter()
            .map(|(idx, frame)| {
                if ((idx - 1) as usize) % interval == 0 {
                    Ok(Some(detector.detect(frame)?))
                } else {
                    Ok(None)
                }
            })
            .collect();
        let batch_detections = batch_detections?;

        // 按帧顺序更新追踪器与向前追溯缓冲队列，收集本批次可出队的就绪帧
        let mut last_active_count = 0usize;
        let mut ready_batch: Vec<BufferedFrame> = Vec::with_capacity(batch.len());

        for ((_idx, frame), maybe_det) in batch.into_iter().zip(batch_detections) {
            let active_faces = match maybe_det {
                Some(raw_detections) => tracker.update(raw_detections),
                None => tracker.current_faces(),
            };
            last_active_count = active_faces.len();

            backfill_faces(&mut history, &active_faces, 0.25);

            history.push_back(BufferedFrame {
                image: frame,
                faces: active_faces,
            });

            if history.len() > pre_frames {
                let ready = history.pop_front().unwrap();
                total_faces_detected += ready.faces.len() as u64;
                ready_batch.push(ready);
            }
        }

        // 使用 rayon 多核并行对就绪帧执行 Alpha 头像贴图合成
        ready_batch.par_iter_mut().for_each(|ready| {
            if !ready.faces.is_empty() {
                overlay.apply(&mut ready.image, &ready.faces);
            }
        });

        // 按顺序送入独立编码写入线程
        for ready in ready_batch {
            enc_tx
                .send(ready.image)
                .map_err(|_| anyhow::anyhow!("编码写入线程已意外退出"))?;
        }

        let elapsed = start_time.elapsed().as_secs_f32().max(0.001);
        let fps = frame_count as f32 / elapsed;
        if let Some(total) = meta.total_frames {
            print!(
                "\r处理进度: {}/{} 帧 ({:.1}%), 速度: {:.1} fps, 当前帧人脸数: {}   ",
                frame_count,
                total,
                (frame_count as f32 / total as f32) * 100.0,
                fps,
                last_active_count
            );
        } else {
            print!(
                "\r已处理: {} 帧, 速度: {:.1} fps, 当前帧人脸数: {}   ",
                frame_count,
                fps,
                last_active_count
            );
        }
        let _ = std::io::stdout().flush();
    }

    // 视频流读取完毕，将缓冲队列中剩余的帧并行打码并送入编码线程
    let mut remaining: Vec<BufferedFrame> = history.into_iter().collect();
    for item in &remaining {
        total_faces_detected += item.faces.len() as u64;
    }
    remaining.par_iter_mut().for_each(|ready| {
        if !ready.faces.is_empty() {
            overlay.apply(&mut ready.image, &ready.faces);
        }
    });
    for ready in remaining {
        enc_tx
            .send(ready.image)
            .map_err(|_| anyhow::anyhow!("编码写入线程已意外退出"))?;
    }

    drop(enc_tx);

    let _ = reader_handle.join();
    writer_handle
        .join()
        .map_err(|_| anyhow::anyhow!("等待编码写入线程结束失败"))??;

    println!();

    let dec_status = decoder.wait().context("等待 ffmpeg 解码进程结束失败")?;
    let enc_status = encoder.wait().context("等待 ffmpeg 编码进程结束失败")?;

    ensure!(dec_status.success(), "ffmpeg 解码过程异常退出");
    ensure!(enc_status.success(), "ffmpeg 编码过程异常退出");

    let elapsed = start_time.elapsed().as_secs_f32();
    let avg_fps = if elapsed > 0.0 {
        frame_count as f32 / elapsed
    } else {
        0.0
    };
    println!(
        "处理完成！共处理 {} 帧，累计替换人脸 {} 人次，耗时 {:.2}s (平均 {:.1} fps)，输出文件: {}",
        frame_count,
        total_faces_detected,
        elapsed,
        avg_fps,
        output.display()
    );

    Ok(())
}

fn read_exact_or_eof<R: Read>(reader: &mut R, mut buf: &mut [u8]) -> Result<bool> {
    let total = buf.len();
    let mut read_len = 0;
    while !buf.is_empty() {
        match reader.read(buf) {
            Ok(0) => {
                if read_len == 0 {
                    return Ok(false);
                }
                bail!(
                    "视频流意外截断（期望 {} 字节，实际读取 {} 字节）",
                    total,
                    read_len
                );
            }
            Ok(n) => {
                read_len += n;
                buf = &mut buf[n..];
            }
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e).context("读取解码视频流失败"),
        }
    }
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backfill_faces_fills_unmasked_previous_frames() {
        let mut history = VecDeque::new();
        let existing_face = FaceBox {
            x1: 0.1,
            y1: 0.1,
            x2: 0.3,
            y2: 0.3,
            score: 0.9,
        };

        // 第 0 帧：没有任何打码
        history.push_back(BufferedFrame {
            image: RgbImage::new(10, 10),
            faces: vec![],
        });
        // 第 1 帧：已有同一个位置的人脸打码
        history.push_back(BufferedFrame {
            image: RgbImage::new(10, 10),
            faces: vec![existing_face],
        });

        // 当前帧检测到 (0.12, 0.12, 0.32, 0.32) 的人脸
        let new_detected = FaceBox {
            x1: 0.12,
            y1: 0.12,
            x2: 0.32,
            y2: 0.32,
            score: 0.88,
        };
        backfill_faces(&mut history, &[new_detected], 0.25);

        // 第 0 帧原本未打码，应被回填补上该位置的人脸框
        assert_eq!(history[0].faces.len(), 1);
        assert_eq!(history[0].faces[0], new_detected);
        // 第 1 帧原本已有重叠打码，不应重复叠加
        assert_eq!(history[1].faces.len(), 1);
        assert_eq!(history[1].faces[0], existing_face);
    }
}

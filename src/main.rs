mod detector;
mod overlay;
mod video;

use anyhow::{Result, ensure};
use clap::Parser;
use detector::{FaceDetector, FaceTracker};
use overlay::AvatarOverlay;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(
    name = "clear_face",
    version,
    about = "读取 MP4 视频并将所有人脸替换为指定头像图片（如葫芦娃头像），同时对人声进行隐私变声"
)]
struct Args {
    /// 输入 MP4 视频文件路径
    #[arg(short, long)]
    input: PathBuf,

    /// 用于替换人脸的头像图片路径（支持 PNG 透明通道、JPG、WebP）
    #[arg(short, long)]
    avatar: PathBuf,

    /// 输出 MP4 视频文件路径
    #[arg(short, long)]
    output: PathBuf,

    /// 头像相对于人脸检测框的缩放比例（默认 1.3，可完整遮挡头发与下巴）
    #[arg(short, long, default_value_t = 1.3)]
    scale: f32,

    /// 人脸检测置信度阈值 (0.0 ~ 1.0，默认 0.7)
    #[arg(short, long, default_value_t = 0.7)]
    threshold: f32,

    /// NMS 非极大值抑制 IoU 阈值 (0.0 ~ 1.0)
    #[arg(long, default_value_t = 0.3)]
    iou_threshold: f32,

    /// 短暂漏检向后保持帧数（防止人物快速转头或模糊时头像闪烁，设为 0 可关闭）
    #[arg(long, default_value_t = 5)]
    hold_frames: usize,

    /// 检测到人脸后向前追溯补码的秒数（默认 0.08 秒，将前序未打码帧的相同位置一并打码）
    #[arg(long, default_value_t = 0.08)]
    pre_seconds: f32,

    /// 每隔多少帧执行一次神经网络检测（默认 2，中间帧复用追踪位置以加速；设为 1 则逐帧检测）
    #[arg(long, default_value_t = 2)]
    detect_interval: usize,

    /// 使用轻量级 320x240 检测模型（速度再快约 3.5 倍，适合非密集远景小脸的普通视频）
    #[arg(long, default_value_t = false)]
    fast_model: bool,

    /// 禁用硬件加速，强制使用 CPU 软编解码 (libx264)
    #[arg(long, default_value_t = false)]
    sw_codec: bool,

    /// 人声隐私变声音高系数（默认 1.38 为卡通/高音色，可设为 0.72 变为低沉男声）
    #[arg(long, default_value_t = 1.38)]
    voice_pitch: f32,

    /// 禁用人声变声，直接无损复制原视频音频
    #[arg(long, default_value_t = false)]
    keep_audio: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();

    ensure!(
        args.input.exists(),
        "输入视频文件不存在: {}",
        args.input.display()
    );
    ensure!(
        args.avatar.exists(),
        "头像图片文件不存在: {}",
        args.avatar.display()
    );
    ensure!(
        (0.05..=1.0).contains(&args.threshold),
        "置信度阈值 threshold 需在 0.05 ~ 1.0 之间"
    );
    ensure!(
        args.pre_seconds >= 0.0,
        "向前追溯秒数 pre_seconds 不能为负数"
    );
    ensure!(
        args.detect_interval >= 1,
        "检测间隔帧数 detect_interval 必须大于等于 1"
    );
    ensure!(
        (0.3..=3.0).contains(&args.voice_pitch),
        "变声音高系数 voice_pitch 需在 0.3 ~ 3.0 之间"
    );

    println!(
        "正在初始化 UltraFace 人脸检测模型 ({})...",
        if args.fast_model {
            "RFB-320 极速模式"
        } else {
            "RFB-640 高清模式"
        }
    );
    let detector = FaceDetector::new(args.threshold, args.iou_threshold, args.fast_model)?;
    let mut tracker = FaceTracker::new(args.hold_frames);
    let overlay = AvatarOverlay::from_path(&args.avatar, args.scale)?;

    video::process_video(
        &args.input,
        &args.output,
        &detector,
        &mut tracker,
        &overlay,
        args.pre_seconds,
        args.detect_interval,
        args.sw_codec,
        args.keep_audio,
        args.voice_pitch,
    )?;

    Ok(())
}

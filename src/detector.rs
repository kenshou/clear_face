use anyhow::{Context, Result};
use image::RgbImage;
use std::io::Cursor;
use tract_onnx::prelude::*;

const EMBEDDED_MODEL_640: &[u8] = include_bytes!("../models/version-RFB-640.onnx");
const EMBEDDED_MODEL_320: &[u8] = include_bytes!("../models/version-RFB-320.onnx");

type OnnxModel = SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FaceBox {
    /// 归一化坐标 (0.0 ~ 1.0)
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    pub score: f32,
}

impl FaceBox {
    pub fn area(&self) -> f32 {
        (self.x2 - self.x1).max(0.0) * (self.y2 - self.y1).max(0.0)
    }

    pub fn iou(&self, other: &Self) -> f32 {
        let inter_x1 = self.x1.max(other.x1);
        let inter_y1 = self.y1.max(other.y1);
        let inter_x2 = self.x2.min(other.x2);
        let inter_y2 = self.y2.min(other.y2);

        let inter_area = (inter_x2 - inter_x1).max(0.0) * (inter_y2 - inter_y1).max(0.0);
        if inter_area <= 0.0 {
            return 0.0;
        }
        let union_area = self.area() + other.area() - inter_area;
        if union_area <= 0.0 {
            0.0
        } else {
            inter_area / union_area
        }
    }

    pub fn lerp(&self, target: &Self, alpha: f32) -> Self {
        Self {
            x1: self.x1 + (target.x1 - self.x1) * alpha,
            y1: self.y1 + (target.y1 - self.y1) * alpha,
            x2: self.x2 + (target.x2 - self.x2) * alpha,
            y2: self.y2 + (target.y2 - self.y2) * alpha,
            score: target.score,
        }
    }
}

pub struct FaceDetector {
    model: OnnxModel,
    model_width: u32,
    model_height: u32,
    score_threshold: f32,
    iou_threshold: f32,
}

impl FaceDetector {
    pub fn new(score_threshold: f32, iou_threshold: f32, fast_model: bool) -> Result<Self> {
        let (model_bytes, model_width, model_height) = if fast_model {
            (EMBEDDED_MODEL_320, 320u32, 240u32)
        } else {
            (EMBEDDED_MODEL_640, 640u32, 480u32)
        };

        let mut cursor = Cursor::new(model_bytes);
        let model = tract_onnx::onnx()
            .model_for_read(&mut cursor)
            .context("解析内置 UltraFace ONNX 模型失败")?
            .with_input_fact(
                0,
                InferenceFact::dt_shape(
                    f32::datum_type(),
                    tvec![1, 3, model_height as usize, model_width as usize],
                ),
            )
            .context("设置模型输入维度失败")?
            .into_optimized()
            .context("优化 ONNX 计算图失败")?
            .into_runnable()
            .context("构建可执行 ONNX 模型失败")?;

        Ok(Self {
            model,
            model_width,
            model_height,
            score_threshold,
            iou_threshold,
        })
    }

    pub fn detect(&self, frame: &RgbImage) -> Result<Vec<FaceBox>> {
        let src_w = frame.width() as usize;
        let src_h = frame.height() as usize;
        let dst_w = self.model_width as usize;
        let dst_h = self.model_height as usize;
        let hw = dst_w * dst_h;

        let mut input_data = vec![0.0f32; 3 * hw];
        let raw = frame.as_raw();
        let inv_128 = 1.0f32 / 128.0;

        // 单次直接采样并归一化，避免分配中间缩放图像与双线性插值开销
        let x_offsets: Vec<usize> = (0..dst_w)
            .map(|dx| ((dx * src_w) / dst_w).min(src_w.saturating_sub(1)) * 3)
            .collect();

        for dy in 0..dst_h {
            let sy = ((dy * src_h) / dst_h).min(src_h.saturating_sub(1));
            let src_row = sy * src_w * 3;
            let dst_row = dy * dst_w;
            for (dx, &sx_byte) in x_offsets.iter().enumerate() {
                let px = src_row + sx_byte;
                let dst_idx = dst_row + dx;
                input_data[dst_idx] = (raw[px] as f32 - 127.0) * inv_128;
                input_data[hw + dst_idx] = (raw[px + 1] as f32 - 127.0) * inv_128;
                input_data[2 * hw + dst_idx] = (raw[px + 2] as f32 - 127.0) * inv_128;
            }
        }

        let input_tensor = tract_ndarray::Array4::from_shape_vec(
            (1, 3, dst_h, dst_w),
            input_data,
        )?
        .into_tensor();

        let outputs = self.model.run(tvec!(input_tensor.into()))?;
        let scores = outputs[0].to_array_view::<f32>()?;
        let boxes = outputs[1].to_array_view::<f32>()?;

        let num_anchors = scores.shape()[1];
        let mut candidates = Vec::new();

        for i in 0..num_anchors {
            let score = scores[[0, i, 1]];
            if score >= self.score_threshold {
                let x1 = boxes[[0, i, 0]].clamp(0.0, 1.0);
                let y1 = boxes[[0, i, 1]].clamp(0.0, 1.0);
                let x2 = boxes[[0, i, 2]].clamp(0.0, 1.0);
                let y2 = boxes[[0, i, 3]].clamp(0.0, 1.0);
                if x2 > x1 && y2 > y1 {
                    candidates.push(FaceBox {
                        x1,
                        y1,
                        x2,
                        y2,
                        score,
                    });
                }
            }
        }

        Ok(nms(candidates, self.iou_threshold))
    }
}

pub fn nms(mut boxes: Vec<FaceBox>, iou_threshold: f32) -> Vec<FaceBox> {
    boxes.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut kept: Vec<FaceBox> = Vec::new();
    for candidate in boxes {
        let overlapped = kept
            .iter()
            .any(|existing| existing.iou(&candidate) > iou_threshold);
        if !overlapped {
            kept.push(candidate);
        }
    }
    kept
}

#[derive(Debug, Clone)]
struct TrackedFace {
    bbox: FaceBox,
    missed_frames: usize,
}

/// 帧间人脸平滑与短暂漏检保持器，避免单帧漏检导致头像闪烁
pub struct FaceTracker {
    tracks: Vec<TrackedFace>,
    max_hold_frames: usize,
    match_iou_threshold: f32,
    smooth_alpha: f32,
}

impl FaceTracker {
    pub fn new(max_hold_frames: usize) -> Self {
        Self {
            tracks: Vec::new(),
            max_hold_frames,
            match_iou_threshold: 0.25,
            smooth_alpha: 0.65,
        }
    }

    /// 隔帧跳过检测时，直接返回当前活跃的追踪人脸框（不增加漏检计数）
    pub fn current_faces(&self) -> Vec<FaceBox> {
        self.tracks.iter().map(|t| t.bbox).collect()
    }

    pub fn update(&mut self, detections: Vec<FaceBox>) -> Vec<FaceBox> {
        if self.max_hold_frames == 0 {
            self.tracks = detections
                .iter()
                .map(|&bbox| TrackedFace {
                    bbox,
                    missed_frames: 0,
                })
                .collect();
            return detections;
        }

        let mut matched_det = vec![false; detections.len()];
        let mut next_tracks = Vec::new();

        for track in &self.tracks {
            let mut best_idx = None;
            let mut best_iou = self.match_iou_threshold;

            for (idx, det) in detections.iter().enumerate() {
                if matched_det[idx] {
                    continue;
                }
                let iou = track.bbox.iou(det);
                if iou > best_iou {
                    best_iou = iou;
                    best_idx = Some(idx);
                }
            }

            if let Some(idx) = best_idx {
                matched_det[idx] = true;
                next_tracks.push(TrackedFace {
                    bbox: track.bbox.lerp(&detections[idx], self.smooth_alpha),
                    missed_frames: 0,
                });
            } else if track.missed_frames < self.max_hold_frames {
                next_tracks.push(TrackedFace {
                    bbox: track.bbox,
                    missed_frames: track.missed_frames + 1,
                });
            }
        }

        for (idx, det) in detections.into_iter().enumerate() {
            if !matched_det[idx] {
                next_tracks.push(TrackedFace {
                    bbox: det,
                    missed_frames: 0,
                });
            }
        }

        self.tracks = next_tracks;
        self.current_faces()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nms_removes_overlapping_boxes() {
        let boxes = vec![
            FaceBox {
                x1: 0.1,
                y1: 0.1,
                x2: 0.3,
                y2: 0.3,
                score: 0.9,
            },
            FaceBox {
                x1: 0.11,
                y1: 0.11,
                x2: 0.31,
                y2: 0.31,
                score: 0.8,
            },
            FaceBox {
                x1: 0.6,
                y1: 0.6,
                x2: 0.8,
                y2: 0.8,
                score: 0.85,
            },
        ];

        let kept = nms(boxes, 0.3);
        assert_eq!(kept.len(), 2);
        assert!((kept[0].score - 0.9).abs() < 1e-5);
        assert!((kept[1].score - 0.85).abs() < 1e-5);
    }

    #[test]
    fn test_tracker_holds_missed_frames() {
        let mut tracker = FaceTracker::new(2);
        let face = FaceBox {
            x1: 0.2,
            y1: 0.2,
            x2: 0.4,
            y2: 0.4,
            score: 0.95,
        };

        assert_eq!(tracker.update(vec![face]).len(), 1);
        assert_eq!(tracker.current_faces().len(), 1);
        // 漏检第 1 帧、第 2 帧保持
        assert_eq!(tracker.update(vec![]).len(), 1);
        assert_eq!(tracker.update(vec![]).len(), 1);
        // 超过保持帧数后移除
        assert_eq!(tracker.update(vec![]).len(), 0);
    }
}

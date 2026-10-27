use crate::detector::FaceBox;
use anyhow::{Context, Result, ensure};
use image::{RgbaImage, RgbImage, imageops::FilterType};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Mutex;

pub struct AvatarOverlay {
    avatar: RgbaImage,
    scale: f32,
    cache: Mutex<HashMap<(u32, u32), RgbaImage>>,
}

impl AvatarOverlay {
    pub fn from_path(path: &Path, scale: f32) -> Result<Self> {
        ensure!(scale > 0.0, "缩放比例 scale 必须大于 0");
        let img = image::open(path)
            .with_context(|| format!("无法读取头像图片: {}", path.display()))?
            .to_rgba8();
        ensure!(
            img.width() > 0 && img.height() > 0,
            "头像图片尺寸无效"
        );
        Ok(Self {
            avatar: img,
            scale,
            cache: Mutex::new(HashMap::new()),
        })
    }

    #[cfg(test)]
    pub fn from_rgba(avatar: RgbaImage, scale: f32) -> Self {
        Self {
            avatar,
            scale,
            cache: Mutex::new(HashMap::new()),
        }
    }

    pub fn apply(&self, frame: &mut RgbImage, faces: &[FaceBox]) {
        let frame_w = frame.width() as f32;
        let frame_h = frame.height() as f32;
        let avatar_w = self.avatar.width() as f32;
        let avatar_h = self.avatar.height() as f32;

        for face in faces {
            let box_w = (face.x2 - face.x1) * frame_w;
            let box_h = (face.y2 - face.y1) * frame_h;
            if box_w < 2.0 || box_h < 2.0 {
                continue;
            }

            let cx = (face.x1 + face.x2) * 0.5 * frame_w;
            let cy = (face.y1 + face.y2) * 0.5 * frame_h;

            let target_box_w = box_w * self.scale;
            let target_box_h = box_h * self.scale;

            // 保持头像原始宽高比，同时按 2 像素粒度对齐以提高缩放缓存命中率
            let ratio = (target_box_w / avatar_w).max(target_box_h / avatar_h);
            let raw_w = (avatar_w * ratio).round().max(2.0) as u32;
            let raw_h = (avatar_h * ratio).round().max(2.0) as u32;
            let target_w = (raw_w + 1) & !1;
            let target_h = (raw_h + 1) & !1;

            let top_left_x = (cx - target_w as f32 * 0.5).round() as i64;
            let top_left_y = (cy - target_h as f32 * 0.5).round() as i64;

            let mut cache = self.cache.lock().unwrap();
            if cache.len() > 64 {
                cache.clear();
            }
            let resized_avatar = cache.entry((target_w, target_h)).or_insert_with(|| {
                image::imageops::resize(
                    &self.avatar,
                    target_w,
                    target_h,
                    FilterType::Triangle,
                )
            });

            blend_rgba_onto_rgb(frame, resized_avatar, top_left_x, top_left_y);
        }
    }
}

/// 将带 Alpha 通道的 RGBA 图像混合绘制到 RGB 帧上，自动裁剪超出画面边界的像素
fn blend_rgba_onto_rgb(
    dst: &mut RgbImage,
    src: &RgbaImage,
    offset_x: i64,
    offset_y: i64,
) {
    let dst_w = dst.width() as i64;
    let dst_h = dst.height() as i64;
    let src_w = src.width() as i64;
    let src_h = src.height() as i64;

    let start_x = 0.max(-offset_x);
    let start_y = 0.max(-offset_y);
    let end_x = src_w.min(dst_w - offset_x);
    let end_y = src_h.min(dst_h - offset_y);

    if start_x >= end_x || start_y >= end_y {
        return;
    }

    let dst_stride = (dst_w as usize) * 3;
    let src_stride = (src_w as usize) * 4;
    let dst_raw = dst.as_mut();
    let src_raw = src.as_raw();

    for sy in start_y..end_y {
        let dy = (offset_y + sy) as usize;
        let dst_row_start = dy * dst_stride;
        let src_row_start = (sy as usize) * src_stride;

        for sx in start_x..end_x {
            let s_idx = src_row_start + (sx as usize) * 4;
            let alpha = src_raw[s_idx + 3] as u32;
            if alpha == 0 {
                continue;
            }

            let dx = (offset_x + sx) as usize;
            let d_idx = dst_row_start + dx * 3;

            if alpha == 255 {
                dst_raw[d_idx] = src_raw[s_idx];
                dst_raw[d_idx + 1] = src_raw[s_idx + 1];
                dst_raw[d_idx + 2] = src_raw[s_idx + 2];
            } else {
                let inv_alpha = 255 - alpha;
                dst_raw[d_idx] =
                    ((src_raw[s_idx] as u32 * alpha + dst_raw[d_idx] as u32 * inv_alpha) / 255)
                        as u8;
                dst_raw[d_idx + 1] = ((src_raw[s_idx + 1] as u32 * alpha
                    + dst_raw[d_idx + 1] as u32 * inv_alpha)
                    / 255) as u8;
                dst_raw[d_idx + 2] = ((src_raw[s_idx + 2] as u32 * alpha
                    + dst_raw[d_idx + 2] as u32 * inv_alpha)
                    / 255) as u8;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Rgb, Rgba};

    #[test]
    fn test_overlay_blending_and_out_of_bounds_clipping() {
        let mut frame = RgbImage::from_pixel(100, 100, Rgb([10, 20, 30]));
        let avatar = RgbaImage::from_pixel(20, 20, Rgba([200, 100, 50, 255]));
        let overlay = AvatarOverlay::from_rgba(avatar, 1.0);

        // 放置一个靠近左上角边缘的人脸框（触发负坐标裁剪）
        let faces = vec![FaceBox {
            x1: 0.0,
            y1: 0.0,
            x2: 0.1,
            y2: 0.1,
            score: 0.99,
        }];

        overlay.apply(&mut frame, &faces);

        // 人脸中心 (5, 5) 应该被替换为头像颜色
        assert_eq!(frame.get_pixel(5, 5).0, [200, 100, 50]);
        // 远离人脸的右下角保持原背景色
        assert_eq!(frame.get_pixel(90, 90).0, [10, 20, 30]);
    }
}

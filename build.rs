use std::fs;
use std::path::Path;
use std::process::Command;

const MODELS: &[(&str, &str)] = &[
    (
        "models/version-RFB-640.onnx",
        "https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB/raw/master/models/onnx/version-RFB-640.onnx",
    ),
    (
        "models/version-RFB-320.onnx",
        "https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB/raw/master/models/onnx/version-RFB-320.onnx",
    ),
];

fn main() {
    println!("cargo:rerun-if-changed=models/version-RFB-640.onnx");
    println!("cargo:rerun-if-changed=models/version-RFB-320.onnx");

    fs::create_dir_all("models").expect("无法创建 models 目录");

    for (path_str, url) in MODELS {
        let path = Path::new(path_str);
        if path.exists() {
            continue;
        }
        println!("cargo:warning=正在下载人脸检测模型: {}", path_str);
        let status = Command::new("curl")
            .args(["-fL", url, "-o", path_str])
            .status()
            .expect("无法调用 curl 下载模型文件，请手动下载至 models/ 目录");
        assert!(status.success(), "下载模型文件失败: {}", url);
    }
}

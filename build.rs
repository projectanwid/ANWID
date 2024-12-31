use std::env;
use std::path::Path;
use std::process::Command;

fn find_msvc_tools() -> Option<String> {
    let vswhere_path = "C:\\Program Files (x86)\\Microsoft Visual Studio\\Installer\\vswhere.exe";

    if Path::new(vswhere_path).exists() {
        let output = Command::new(vswhere_path)
            .args(&["-latest", "-property", "installationPath"])
            .output()
            .ok()?;

        if output.status.success() {
            if let Ok(vs_path) = String::from_utf8(output.stdout) {
                let vs_path = vs_path.trim();
                let msvc_path = format!("{}\\VC\\Tools\\MSVC", vs_path);

                if let Ok(entries) = std::fs::read_dir(&msvc_path) {
                    if let Some(latest_version) = entries
                        .filter_map(|e| e.ok())
                        .filter(|e| e.path().is_dir())
                        .max_by(|a, b| a.file_name().cmp(&b.file_name()))
                    {
                        let cl_path = format!(
                            "{}\\{}\\bin\\Hostx64\\x64",
                            msvc_path,
                            latest_version.file_name().to_string_lossy()
                        );
                        if Path::new(&format!("{}\\cl.exe", cl_path)).exists() {
                            return Some(cl_path);
                        }
                    }
                }
            }
        }
    }
    None
}

fn setup_msvc() -> Result<(), String> {
    let msvc_path = find_msvc_tools()
        .ok_or_else(|| "Could not find Visual Studio C++ build tools".to_string())?;

    let old_path = env::var("PATH").map_err(|e| format!("Failed to get PATH: {}", e))?;
    let new_path = format!("{};{}", msvc_path, old_path);
    env::set_var("PATH", new_path);

    env::set_var("INCLUDE", format!("{}\\..\\..\\..\\include", msvc_path));
    env::set_var("LIB", format!("{}\\..\\..\\..\\lib\\x64", msvc_path));
    Ok(())
}

fn find_cuda_windows() -> Option<String> {
    if let Ok(cuda_path) = env::var("CUDA_PATH") {
        if Path::new(&cuda_path).exists() {
            return Some(cuda_path);
        }
    }

    let program_files_paths = [
        env::var("ProgramFiles").unwrap_or("C:\\Program Files".to_string()),
        env::var("ProgramFiles(x86)").unwrap_or("C:\\Program Files (x86)".to_string()),
        "C:\\Program Files".to_string(),
    ];

    let cuda_versions = [
        "v12.6", "v12.3", "v12.2", "v12.1", "v12.0", "v11.8", "v11.7", "v11.6",
    ];

    for program_files in program_files_paths.iter() {
        for version in cuda_versions.iter() {
            let cuda_path = format!(
                "{}\\NVIDIA GPU Computing Toolkit\\CUDA\\{}",
                program_files, version
            );
            if Path::new(&cuda_path).exists() {
                let nvcc_path = format!("{}\\bin\\nvcc.exe", cuda_path);
                if Path::new(&nvcc_path).exists() {
                    return Some(cuda_path);
                }
            }
        }
    }

    None
}

//noinspection ALL
fn compile_cuda_kernel() -> std::io::Result<()> {
    if cfg!(windows) {
        setup_msvc().map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
    }

    let cuda_path = find_cuda_windows()
        .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::NotFound, "CUDA installation not found"))?;

    let build_dir = Path::new("build");
    std::fs::create_dir_all(build_dir)?;

    let (lib_name, lib_ext) = if cfg!(windows) {
        ("anwid_kernel", "lib")
    } else {
        ("libanwid_kernel", "a")
    };

    let _output_file = format!("build/{}.{}", lib_name, lib_ext);
    let nvcc_path = format!("{}\\bin\\nvcc.exe", cuda_path);

    let mut nvcc_command = Command::new(&nvcc_path);
    nvcc_command
        .args(&[
            "-O3",
            "--use_fast_math",
            "-Xcompiler",
            "/MD",
            "-Xptxas",
            "-v",
            "--compiler-options",
            "/W4",
            "--compiler-options",
            "/WX",
            "--compiler-options",
            "/EHsc",
            "-gencode",
            "arch=compute_50,code=sm_50",
            "-gencode",
            "arch=compute_60,code=sm_60",
            "-gencode",
            "arch=compute_70,code=sm_70",
            "-gencode",
            "arch=compute_75,code=sm_75",
            "-gencode",
            "arch=compute_80,code=sm_80",
            "-gencode",
            "arch=compute_86,code=sm_86",
            "-gencode",
            "arch=compute_89,code=sm_89",
            "--default-stream",
            "per-thread",
            "-Xcompiler",
            "/Zc:preprocessor",
            "-c",
            "anwid_kernel.cu",
            "-o",
            "build/anwid_kernel.obj"
        ]);

    let output = nvcc_command.output()?;
    if !output.status.success() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::Other,
            format!(
                "NVCC compilation failed:\nstdout: {}\nstderr: {}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            ),
        ));
    }

    let lib_command = Command::new("lib.exe")
        .args(&[
            "/OUT:build/anwid_kernel.lib",
            "build/anwid_kernel.obj",
        ])
        .output()?;

    if !lib_command.status.success() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::Other,
            format!(
                "Library creation failed:\nstdout: {}\nstderr: {}",
                String::from_utf8_lossy(&lib_command.stdout),
                String::from_utf8_lossy(&lib_command.stderr)
            ),
        ));
    }

    println!("cargo:rerun-if-changed=anwid_kernel.cu");
    println!("cargo:rustc-link-search=native=build");
    println!("cargo:rustc-link-lib=static=anwid_kernel");
    
    // Add CUDA library path and link against cudart
    println!("cargo:rustc-link-search=native={}/lib/x64", cuda_path);
    println!("cargo:rustc-link-lib=cudart");
    Ok(())
}

fn main() {
    if let Err(e) = compile_cuda_kernel() {
        eprintln!("Failed to compile CUDA kernel: {}", e);
        std::process::exit(1);
    }
}

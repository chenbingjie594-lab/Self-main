Set-Location "C:\MyFile\Study\代码\Self-main"

$Yolo = "C:\Users\cbj\miniconda3\envs\yolov11\Scripts\yolo.exe"
$DataRoot = "Y:\dataset\GLRR_detection_hgrr_v2_unique40_val20_plus_black35_s42"
$LogDir = "C:\MyFile\Study\代码\Self-main\logs\yolo_plus_black35"
$Project = "C:\MyFile\Study\代码\Self-main\runs\yolo_plus_black35"

$Groups = @(
    "real_only",
    "real_baseline",
    "real_carf"
)

foreach ($Group in $Groups) {
    $Time = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "[$Time] START $Group" |
        Out-File "$LogDir\train_all.log" -Append -Encoding utf8

    & $Yolo detect train `
        model="pretrained\yolo11s.pt" `
        data="$DataRoot\$Group.yaml" `
        epochs=150 `
        imgsz=1536 `
        batch=1 `
        rect=False `
        device=0 `
        workers=4 `
        seed=42 `
        deterministic=True `
        patience=40 `
        project="$Project" `
        name="${Group}_s42" `
        exist_ok=False `
        2>&1 |
        Tee-Object -FilePath "$LogDir\${Group}.log" -Append

    $ExitCode = $LASTEXITCODE
    $Time = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "[$Time] END $Group exit_code=$ExitCode" |
        Out-File "$LogDir\train_all.log" -Append -Encoding utf8

    if ($ExitCode -ne 0) {
        "Training failed at $Group" |
            Out-File "$LogDir\train_all.log" -Append -Encoding utf8
        exit $ExitCode
    }
}

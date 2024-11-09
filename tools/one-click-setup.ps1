# Function to check command existence
function Check-Command {
    param (
        [string]$command
    )
    return (Get-Command $command -ErrorAction SilentlyContinue)
}

# Check if Python is installed
$pythonPath = Check-Command -command "python"

if (-not $pythonPath) {
    Write-Host "Python is not installed. Exiting script." -ForegroundColor Red
    return
}

# Get Python version
$pythonVersion = python --version 2>&1
$versionMatch = [regex]::Match($pythonVersion, 'Python (\d+)\.(\d+)\.(\d+)')

if ($versionMatch.Success) {
    $majorVersion = [int]$versionMatch.Groups[1].Value
    $minorVersion = [int]$versionMatch.Groups[2].Value
    $patchVersion = [int]$versionMatch.Groups[3].Value

    # Check if version is between 3.9 and 3.12
    if ($majorVersion -eq 3 -and ($minorVersion -ge 9 -and $minorVersion -le 12)) {
        Write-Host "Python version is $($majorVersion).$($minorVersion).$($patchVersion)" -ForegroundColor Cyan
    } else {
        Write-Host "Pytorch is only supported on Python versions 3.9-3.12. Please consult your administration." -ForegroundColor Red
        return
    }
} else {
    Write-Host "Failed to retrieve Python version." -ForegroundColor Red
    return
}

# Check if GPU is available
$gpuInfo = Check-Command -command "nvidia-smi"

if (-not $gpuInfo) {
    Write-Host "GPU not detected. Model performance may be impacted." -ForegroundColor Red
    $userResponse = Read-Host "Do you want to continue? (Y/N)"
    if ($userResponse -notin @('Y', 'y')) {
        return
    }
}

# Get CUDA version if GPU is available
if ($gpuInfo) {
    $output = & nvidia-smi
    $output = $output -join " "
    if ($output -match 'CUDA Version:\s+(\d+\.\d+)') {
        $cudaVersion = $matches[1]
        Write-Host "CUDA Version: $cudaVersion" -ForegroundColor Cyan
    } else {
        Write-Host "CUDA Version not found." -ForegroundColor Red
        return
    }
}

# Check if pip is installed
$pipPath = Check-Command -command "pip"

# Check if pip is installed
if ($pipPath) {
    Write-Host "pip is installed" -ForegroundColor Cyan
} else {
    Write-Host "pip is not installed." -ForegroundColor Red
    return
}

# Prepare installation commands
$installPytorch = if ($cudaVersion) {
    $cudaVersionModified = $cudaVersion -replace '\.', ''
    "pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu$cudaVersionModified"
} else {
    "pip install torch torchvision torchaudio" # CPU only
}

$installPackages = @(
    $installPytorch,
    "pip install opencv-python",
    "pip install numpy",
    "pip install ultralytics",
    "pip install pyvips"
)

# Install packages
foreach ($command in $installPackages) {
    Write-Host "`nRunning command:`n$command" -ForegroundColor Cyan
    Invoke-Expression $command
}

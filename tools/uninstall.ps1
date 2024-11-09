$installPackages = @(
	"pip uninstall torch torchvision torchaudio",
	"pip uninstall opencv-python",
	"pip uninstall ultralytics",
	"pip uninstall pyvips"
)

foreach ($command in $installPackages) {
    Write-Host "`nRunning command:`n$command" -ForegroundColor Cyan
    Invoke-Expression $command
}
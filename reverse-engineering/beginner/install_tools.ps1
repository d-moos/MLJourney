# Binary Reverse Engineering Beginner Course - Tool Installation Script
# This script attempts to install all required tools and reports failures
# Run as Administrator: powershell -ExecutionPolicy Bypass -File install_tools.ps1

# Check if running as Administrator
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")
if (-not $isAdmin) {
    Write-Host "ERROR: This script must be run as Administrator!" -ForegroundColor Red
    Write-Host "Please run PowerShell as Administrator and try again." -ForegroundColor Yellow
    exit 1
}

Write-Host "========================================" -ForegroundColor Green
Write-Host "Binary Reverse Engineering - Tool Setup" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""

# Track installation results
$installed = @()
$failed = @()

# Helper function to install a tool
function Install-Tool {
    param(
        [string]$ToolName,
        [string]$ChocoPackage,
        [string]$Description,
        [string]$ManualUrl = ""
    )
    
    Write-Host "Installing $ToolName..." -ForegroundColor Cyan
    try {
        choco install $ChocoPackage -y --no-progress 2>&1 | Out-Null
        $installed += @{Name = $ToolName; Description = $Description}
        Write-Host "  ✓ $ToolName installed successfully" -ForegroundColor Green
        return $true
    } catch {
        $failed += @{Name = $ToolName; Description = $Description; Reason = $_.Exception.Message; ManualUrl = $ManualUrl}
        Write-Host "  ✗ Failed to install $ToolName" -ForegroundColor Red
        return $false
    }
}

# Check if Chocolatey is installed
$chocoPath = Get-Command choco -ErrorAction SilentlyContinue
if ($null -eq $chocoPath) {
    Write-Host "Installing Chocolatey package manager..." -ForegroundColor Yellow
    try {
        Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process -Force
        [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
        iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
        Write-Host "✓ Chocolatey installed successfully" -ForegroundColor Green
    } catch {
        Write-Host "✗ Failed to install Chocolatey" -ForegroundColor Red
        Write-Host "  Please install manually from: https://chocolatey.org/install" -ForegroundColor Yellow
        exit 1
    }
}

Write-Host ""
Write-Host "Installing tools..." -ForegroundColor Cyan
Write-Host ""

# Install each tool
Install-Tool "x64dbg" "x64dbg" "Debugger for dynamic analysis" "https://x64dbg.com/" | Out-Null
Install-Tool "Python 3" "python" "Scripting and automation" "https://www.python.org/downloads/" | Out-Null
Install-Tool "Rust" "rustup.install" "Compiler for building binaries" "https://www.rust-lang.org/tools/install" | Out-Null
Install-Tool "PE-bear" "pe-bear" "PE header inspection tool" "https://github.com/hasherezade/pe-bear-releases" | Out-Null
Install-Tool "Process Explorer" "procexp" "Process and module monitoring" "https://learn.microsoft.com/en-us/sysinternals/downloads/process-explorer" | Out-Null
Install-Tool "ProcMon" "procmon" "System activity monitoring" "https://learn.microsoft.com/en-us/sysinternals/downloads/procmon" | Out-Null

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "Installation Summary" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""

# Display installed tools
if ($installed.Count -gt 0) {
    Write-Host "Successfully Installed ($($installed.Count)):" -ForegroundColor Green
    foreach ($tool in $installed) {
        Write-Host "  ✓ $($tool.Name) - $($tool.Description)" -ForegroundColor Green
    }
    Write-Host ""
}

# Display failed tools
if ($failed.Count -gt 0) {
    Write-Host "Failed to Install ($($failed.Count)) - Manual Installation Required:" -ForegroundColor Yellow
    Write-Host ""
    foreach ($tool in $failed) {
        Write-Host "  ✗ $($tool.Name)" -ForegroundColor Red
        Write-Host "    Description: $($tool.Description)" -ForegroundColor White
        if ($tool.ManualUrl) {
            Write-Host "    Download: $($tool.ManualUrl)" -ForegroundColor Cyan
        }
    }
    Write-Host ""
}

# Special note about Binary Ninja
Write-Host "IMPORTANT: Binary Ninja Installation" -ForegroundColor Yellow
Write-Host "Binary Ninja is not available via Chocolatey (requires license)." -ForegroundColor White
Write-Host "Download and install manually from: https://binary.ninja/" -ForegroundColor Cyan
Write-Host "  - Free version available (limited features)" -ForegroundColor White
Write-Host "  - Professional version recommended for full features" -ForegroundColor White
Write-Host ""

# Summary
Write-Host "========================================" -ForegroundColor Green
Write-Host "Next Steps:" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""

if ($failed.Count -eq 0) {
    Write-Host "1. Install Binary Ninja from https://binary.ninja/" -ForegroundColor Cyan
    Write-Host "2. Verify all tools are working" -ForegroundColor Cyan
    Write-Host "3. Start with Lesson 1: Setting up a Safe Reversing Lab" -ForegroundColor Cyan
} else {
    Write-Host "1. Install the following tools manually:" -ForegroundColor Cyan
    foreach ($tool in $failed) {
        Write-Host "   - $($tool.Name): $($tool.ManualUrl)" -ForegroundColor Yellow
    }
    Write-Host "2. Install Binary Ninja from https://binary.ninja/" -ForegroundColor Cyan
    Write-Host "3. Verify all tools are working" -ForegroundColor Cyan
    Write-Host "4. Start with Lesson 1: Setting up a Safe Reversing Lab" -ForegroundColor Cyan
}

Write-Host ""
Write-Host "To verify installations, run:" -ForegroundColor Cyan
Write-Host "  x64dbg --version" -ForegroundColor Gray
Write-Host "  python --version" -ForegroundColor Gray
Write-Host "  rustc --version" -ForegroundColor Gray
Write-Host ""


# Command-line parameters
param(
    [switch]$CreateZip = $false
)

$ErrorActionPreference = "Stop"

$repo_url = "https://github.com/EtalumaSupport/LumaViewPro.git"

$branch = Read-Host -Prompt "Set branch"
$version = Read-Host -Prompt "Set version"
$lvp_base_w_version = "LumaViewPro-$version"

# Load MSI paths from config file if it exists
$script_dir = Split-Path -Parent $PSCommandPath
$config_file = Join-Path -Path $script_dir -ChildPath "config\build_dependencies.json"
$pylon_msi = ""
$corretto_msi = ""
$wix_bal_ext = ""

if (Test-Path $config_file) {
    try {
        $config = Get-Content -Path $config_file -Raw | ConvertFrom-Json
        
        # Check if pylon path is configured and valid
        if ($config.pylon_driver_msi -and (Test-Path $config.pylon_driver_msi)) {
            $pylon_msi = $config.pylon_driver_msi
            Write-Host "Using Pylon driver MSI from config: $pylon_msi"
        } elseif ($config.pylon_driver_msi) {
            Write-Host "Warning: Pylon driver MSI path in config is invalid: $($config.pylon_driver_msi)"
        }
        
        # Check if corretto path is configured and valid
        if ($config.corretto_jdk_msi -and (Test-Path $config.corretto_jdk_msi)) {
            $corretto_msi = $config.corretto_jdk_msi
            Write-Host "Using Corretto JDK MSI from config: $corretto_msi"
        } elseif ($config.corretto_jdk_msi) {
            Write-Host "Warning: Corretto JDK MSI path in config is invalid: $($config.corretto_jdk_msi)"
        }
        
        # Check if WiX BAL extension path is configured and valid
        if ($config.wix_bal_extension -and (Test-Path $config.wix_bal_extension)) {
            $wix_bal_ext = $config.wix_bal_extension
            Write-Host "Using WiX BAL extension from config: $wix_bal_ext"
        } elseif ($config.wix_bal_extension) {
            Write-Host "Warning: WiX BAL extension path in config is invalid: $($config.wix_bal_extension)"
        }
    } catch {
        Write-Host "Warning: Could not read config file: $_"
    }
} else {
    Write-Host "Config file not found at: $config_file"
    Write-Host "You can create this file with pylon_driver_msi and corretto_jdk_msi paths to skip prompts"
}

# Prompt for any missing paths
if (-not $pylon_msi) {
    $pylon_msi = Read-Host -Prompt "Path to Pylon driver MSI (leave blank to skip bundle creation)"
}
if (-not $corretto_msi) {
    $corretto_msi = Read-Host -Prompt "Path to Amazon Corretto JDK MSI (leave blank to skip bundle creation)"
}

$starting_dir = Get-Location
$working_dir = Join-Path -Path $starting_dir -ChildPath "tmp"
$repo_dir = Join-Path -Path $working_dir -ChildPath  "$lvp_base_w_version"
$artifact_dir = Join-Path -Path $working_dir -ChildPath "artifacts"
$exe_artifacts_dir = Join-Path -Path $starting_dir -ChildPath "exe_artifacts"

Write-Host @"
Current Dir:      $starting_dir
Working Dir:      $working_dir
Repo Dir:         $repo_dir
Artifact Dir:     $artifact_dir
Exe Artifacts:    $exe_artifacts_dir
Version:          $version
Create Zips:      $CreateZip
Pylon MSI:        $pylon_msi
Corretto MSI:     $corretto_msi
"@

# Verify WiX Toolset v6 is installed
Write-Host "Verifying WiX Toolset v6 installation..."
try {
    $wix_version = & wix --version 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "WiX command failed"
    }
    Write-Host "WiX Toolset found: $wix_version"
    
    # Check if it's v6.x
    if ($wix_version -notmatch "^v?6\.\d+") {
        Write-Host "Warning: Expected WiX Toolset v6.x but found: $wix_version"
        $continue = Read-Host "Continue anyway? (y/n)"
        if ($continue -ne "y") {
            Exit 1
        }
    }
} catch {
    Write-Host "Error: WiX Toolset v6 not found or not in PATH"
    Write-Host "Please install WiX Toolset v6 from: https://wixtoolset.org/docs/intro/"
    Exit 1
}

if (Test-Path $working_dir) {
    Remove-Item $working_dir -Recurse -Force
}

if (Test-Path $exe_artifacts_dir) {
    Remove-Item $exe_artifacts_dir -Recurse -Force
}

New-Item -Path $working_dir -ItemType Directory | Out-Null
New-Item -Path $repo_dir -ItemType Directory | Out-Null
New-Item -Path $artifact_dir -ItemType Directory | Out-Null
New-Item -Path $exe_artifacts_dir -ItemType Directory | Out-Null

echo "Cloning $repo_url@$branch for release"
git clone --depth 1 --branch $branch $repo_url $repo_dir
Remove-Item "$repo_dir/.git*" -Recurse -Force
Set-Location -Path $repo_dir

$version_in_file = Get-Content -Path "$repo_dir/version.txt" -TotalCount 1
if ($version_in_file -ne $version) {
    Write-Host "version.txt contents do not match supplied version"
    Exit 1
}

echo "Adding license files to top-level"
Copy-Item '.\licenses\*' -Destination '.\' -Force

if ($CreateZip) {
    Set-Location -Path $working_dir
    echo "Creating .zip bundle of source..."
    $compress = @{
        Path = ".\$lvp_base_w_version"
        CompressionLevel = "Optimal"
        DestinationPath =  "$artifact_dir\$lvp_base_w_version-source.zip"
    }
    Compress-Archive @compress

    echo "Creating .tar.gz bundle of source..."
    tar czf "$artifact_dir\$lvp_base_w_version-source.tar.gz" ".\$lvp_base_w_version"
} else {
    echo "Skipping source zip creation (use -CreateZip to enable)"
}

echo "Generating .exe..."
Set-Location -Path $repo_dir
Copy-Item '.\scripts\appBuild\config\lumaviewpro_win_release.spec' '.\lumaviewpro.spec'
pyinstaller --log-level INFO .\lumaviewpro.spec


$orig_output_dir = ".\dist\lumaviewpro"
$new_output_dir = ".\dist\$lvp_base_w_version"

# Note: Encountered access denied issue when trying to use Rename-Item. For now
# make a new directory and copy the contents instead.
echo "Rename output directory"
New-Item -Path $new_output_dir -ItemType Directory | Out-Null
Copy-Item -Path "$orig_output_dir\*" -Destination $new_output_dir -Recurse

# Convert to absolute path now while we're still in the correct directory
$new_output_dir = Resolve-Path -Path $new_output_dir | Select-Object -ExpandProperty Path

if ($CreateZip) {
    echo "Creating .zip bundle of executable..."
    $compress = @{
        Path = $new_output_dir
        CompressionLevel = "Optimal"
        DestinationPath =  "$artifact_dir\$lvp_base_w_version.zip"
    }
    Compress-Archive @compress
} else {
    echo "Skipping executable zip creation (use -CreateZip to enable)"
}

# Copy apache-maven from deps into the build output
echo "Copying apache-maven from deps..."
$script_dir = Split-Path -Parent $PSCommandPath
$deps_dir = Join-Path -Path $script_dir -ChildPath "build_exe\deps"
$maven_source = Join-Path -Path $deps_dir -ChildPath "apache-maven-3.9.8"
$maven_dest = Join-Path -Path $new_output_dir -ChildPath "apache-maven-3.9.8"

if (Test-Path $maven_source) {
    Copy-Item -Path $maven_source -Destination $maven_dest -Recurse -Force
    echo "Apache Maven copied successfully"
} else {
    Write-Host "Warning: Apache Maven not found at $maven_source"
}

# Build MSI using WiX
echo "Building MSI installer with WiX..."
$script_dir = Split-Path -Parent $PSCommandPath
$wix_dir = Join-Path -Path $script_dir -ChildPath "build_exe\wix"
$package_output_dir = Join-Path -Path $exe_artifacts_dir -ChildPath "package"
New-Item -Path $package_output_dir -ItemType Directory -Force | Out-Null

Set-Location -Path $wix_dir

# Build the MSI package
echo "Running: wix build Package.wxs Folders.wxs..."

# Absolute output path
$abs_package_output = Join-Path -Path $package_output_dir -ChildPath "$lvp_base_w_version.msi"

# WiX requires version in x.x.x format (numeric only), extract it from full version
$wix_version = $version
if ($version -match '^(\d+\.\d+\.\d+)') {
    $wix_version = $matches[1]
} else {
    Write-Host "Warning: Version format may not be compatible with WiX: $version"
}

# Debug output
Write-Host "Debug - Install Dir: $new_output_dir"
Write-Host "Debug - Wix Dir: $wix_dir"
Write-Host "Debug - Package Output: $abs_package_output"
Write-Host "Debug - Product Name: $lvp_base_w_version"
Write-Host "Debug - Version (full): $version"
Write-Host "Debug - Version (WiX): $wix_version"

# Build arguments as array to avoid parsing issues
$wixArgs = @(
    'build'
    '-arch'
    'x64'
    '-d'
    "InstallFolderDir=$new_output_dir"
    '-d'
    "ProjectDir=$wix_dir\"
    '-d'
    "ProductName=$lvp_base_w_version"
    '-d'
    "Version=$wix_version"
    '-out'
    $abs_package_output
    'Package.wxs'
    'Folders.wxs'
)

Write-Host "Debug - WiX Args:"
$wixArgs | ForEach-Object { Write-Host "  $_" }

# Try using native command with explicit quoting using --% stop-parsing token
Write-Host "`nAttempting WiX build..."
$wixExe = (Get-Command wix).Source
& $wixExe build -arch x64 `
    -d "InstallFolderDir=$new_output_dir" `
    -d "ProjectDir=$wix_dir\" `
    -d "ProductName=$lvp_base_w_version" `
    -d "Version=$wix_version" `
    -out $abs_package_output `
    Package.wxs Folders.wxs

if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: MSI build failed"
    Set-Location $starting_dir
    Exit 1
}

echo "MSI built successfully: $package_output_dir\$lvp_base_w_version.msi"

# Build Bundle/EXE using WiX if dependencies are provided
if ($pylon_msi -and $corretto_msi) {
    # Validate that the MSI files exist
    if (-not (Test-Path $pylon_msi)) {
        Write-Host "Warning: Pylon MSI not found at: $pylon_msi"
        Write-Host "Skipping bundle creation"
    } elseif (-not (Test-Path $corretto_msi)) {
        Write-Host "Warning: Corretto MSI not found at: $corretto_msi"
        Write-Host "Skipping bundle creation"
    } else {
        echo "Building Bundle installer with WiX..."
        $bundle_output_dir = Join-Path -Path $exe_artifacts_dir -ChildPath "bundle"
        New-Item -Path $bundle_output_dir -ItemType Directory -Force | Out-Null

        echo "Running: wix build Bundle.wxs..."
        
        # Use absolute paths
        $abs_bundle_output = Join-Path -Path $bundle_output_dir -ChildPath "$lvp_base_w_version-setup.exe"
        
        # Determine extension reference method
        $extReference = ""
        
        # First, check deps folder for BAL extension
        $script_dir_for_ext = Split-Path -Parent $PSCommandPath
        $deps_bal_ext = Join-Path -Path $script_dir_for_ext -ChildPath "build_exe\deps\WixToolset.BootstrapperApplications.wixext.dll"
        
        if (Test-Path $deps_bal_ext) {
            Write-Host "Using WiX BAL extension from deps folder: $deps_bal_ext"
            $extReference = $deps_bal_ext
        } elseif ($wix_bal_ext) {
            Write-Host "Using custom WiX BAL extension from config: $wix_bal_ext"
            $extReference = $wix_bal_ext
        } else {
            Write-Host "Using package manager for WiX BAL extension..."
            
            # Check if BAL extension is damaged and repair if needed
            $extList = & wix extension list 2>&1 | Out-String
            
            if ($extList -match "damaged") {
                Write-Host "BAL extension is damaged, removing and reinstalling..."
                & wix extension remove -g WixToolset.Bal.wixext 2>&1 | Write-Host
                Start-Sleep -Seconds 2
            }
            
            # Ensure the BAL extension is installed globally
            Write-Host "Installing WiX BAL extension..."
            $extAddResult = & wix extension add -g WixToolset.Bal.wixext 2>&1
            Write-Host $extAddResult
            
            # Verify installation
            Write-Host "`nInstalled extensions:"
            & wix extension list 2>&1 | Write-Host
            
            $extReference = "WixToolset.Bal.wixext"
        }
        
        # Build using direct command to ensure proper extension loading
        Write-Host "`nAttempting bundle build with BAL extension..."
        $wixExe = (Get-Command wix).Source
        & $wixExe build -arch x64 `
            -ext $extReference `
            -d "LVPInstallFolderDir=$new_output_dir" `
            -d "LVPMsiDir=$abs_package_output" `
            -d "PylonDriverDir=$pylon_msi" `
            -d "CorretoMsiDir=$corretto_msi" `
            -d "ProductName=$lvp_base_w_version" `
            -d "ProductVersion=$wix_version" `
            -out $abs_bundle_output `
            Bundle.wxs

        if ($LASTEXITCODE -ne 0) {
            Write-Host "Error: Bundle build failed"
        } else {
            echo "Bundle built successfully: $bundle_output_dir\$lvp_base_w_version-setup.exe"
        }
    }
} else {
    echo "Skipping bundle creation (Pylon and/or Corretto MSI paths not provided)"
}

Set-Location $starting_dir

echo ""
echo "=========================================="
echo "Build Complete!"
echo "=========================================="
echo "MSI Package: $package_output_dir\$lvp_base_w_version.msi"
$bundle_file = Join-Path -Path $exe_artifacts_dir -ChildPath "bundle\$lvp_base_w_version-setup.exe"
if (Test-Path $bundle_file) {
    echo "Bundle Setup: $bundle_file"
}
if ($CreateZip) {
    echo "Source Zip: $artifact_dir\$lvp_base_w_version-source.zip"
    echo "Source Tar: $artifact_dir\$lvp_base_w_version-source.tar.gz"
    echo "Executable Zip: $artifact_dir\$lvp_base_w_version.zip"
}
echo "=========================================="

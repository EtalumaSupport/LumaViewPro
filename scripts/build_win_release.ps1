$ErrorActionPreference = "Stop"

# $branch = "main"
# $repo_url = "https://github.com/EtalumaSupport/LumaViewPro.git"
$repo_url = "https://github.com/jmcoreymv/LumaViewPro-Private.git"
$branch = "feature/add-windows-exe-gen"

$version = Read-Host -Prompt "Set version"
$lvp_base_w_version = "LumaViewPro-$version"

$starting_dir = Get-Location
$working_dir = ".\tmp"
$repo_dir = "$working_dir\$lvp_base_w_version"
$artifact_dir = "$working_dir\artifacts"

if (Test-Path $working_dir) {
    Remove-Item $working_dir -Recurse -Force
}

New-Item -Path $working_dir -ItemType Directory 
New-Item -Path $repo_dir -ItemType Directory
New-Item -Path $artifact_dir -ItemType Directory

echo "Cloning $repo_url@$branch for release"
git clone --depth 1 --branch $branch $repo_url $repo_dir
Set-Location -Path $repo_dir

# echo "Creating release directory"
# New-Item -Path '.\release' -ItemType Directory

echo "Adding license files to top-level"
Copy-Item '.\licenses\*' -Destination '.\' -Force

echo "Creating .zip bundle of source..."
$compress = @{
    Path = ".\"
    CompressionLevel = "Optimal"
    DestinationPath =  "..\..\$artifact_dir\$lvp_base_w_version-source.zip"
}
Compress-Archive @compress

echo "Generating .exe..."
Copy-Item '.\scripts\config\lumaviewpro_win_release.spec' '.\lumaviewpro.spec'
pyinstaller --log-level INFO .\lumaviewpro.spec

echo "Rename output directory"
$orig_output_dir = ".\dist\lumaviewpro"
$new_output_dir = ".\dist\$lvp_base_w_version"

# Note: Encountered access denied issue when trying to use Rename-Item. For now
# make a new directory and copy the contents instead.
New-Item -Path $new_output_dir -ItemType Directory
Copy-Item -Path "$orig_output_dir\*" -Destination $new_output_dir -Recurse

echo "Creating .zip bundle of executable..."
$compress = @{
    Path = $new_output_dir
    CompressionLevel = "Optimal"
    DestinationPath =  "..\..\$artifact_dir\$lvp_base_w_version.zip"
}
Compress-Archive @compress

Set-Location $starting_dir


$branch = "main"

$working_dir = ".\tmp"
$repo_dir = "$working_dir\repo"
$artifact_dir = "$working_dir\artifacts"



Remove-Item $working_dir -Recurse -Force
New-Item -Path $working_dir -ItemType Directory
New-Item -Path $repo_dir -ItemType Directory
New-Item -Path $artifact_dir -ItemType Directory

$version = Read-Host -Prompt "Set version"
Write-Host "Version: $version"
$lvp_base_w_version = "LumaViewPro-$version"

echo "Cloning $branch for release"
git clone --depth 1 --branch $branch https://github.com/EtalumaSupport/LumaViewPro.git $repo_dir
Set-Location -Path $repo_dir

# echo "Removing old artifacts if present..."
# Remove-Item '.\dist' -Recurse -Force
# Remove-Item '.\build' -Recurse -Force
# Remove-Item '.\release' -Recurse -Force
# Remove-Item '.\LICENSE.*' -Force

echo "Creating release directory"
New-Item -Path '.\release' -ItemType Directory

echo "Adding license files to top-level"
Copy-Item '.\licenses\*' -Destination '.\' -Force

echo "Creating .zip bundle of source..."
$compress = @{
    Path = ".\"
    CompressionLevel = "Optimal"
    DestinationPath =  "..\..\$artifact_dir\$lvp_base_w_version-source.zip"
}
Compress-Archive @compress

echo "Creating .tar.gz bundle of source..."
tar -czf ".\" "..\..\$artifact_dir\$lvp_base_w_version-source.tar.gz"

echo "Generating .exe..."
pyinstaller .\lumaviewpro.spec

echo "Rename output directory"
$orig_output_dir = ".\dist\lumaviewpro"
$new_output_dir = ".\dist\LumaViewPro-$version"
Rename-Item -Path $orig_output_dir -NewName $new_output_dir

echo "Creating .zip bundle of executable..."
$compress = @{
    Path = $new_output_dir
    CompressionLevel = "Optimal"
    DestinationPath =  "..\..\$artifact_dir\$lvp_base_w_version.zip"
}
Compress-Archive @compress


# Windows PowerShell 5.1 or PowerShell 7. Download and verify only; no extraction.
[CmdletBinding()]
param(
    [ValidateSet('GitHub', 'Local')]
    [string]$Source = 'GitHub',
    [string]$Destination = (Join-Path (Get-Location) 'minecraft-final-export-20261021')
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# Publication pins. These must be filled from the completed publication receipt.
$Repository = 'catlover-bot/Minecraft-view-dataset'
$CatalogueName = 'ARTIFACTS.json'
$CatalogueAssetId = '661560760'
$CatalogueSha256 = '740083ce424c9d87d2b988ce915d6acbbc6597a5b92fd92ec34e46145232ac0e'
$ExpectedReleaseId = '419216573'

function Assert-DecimalId {
    param([object]$Value, [string]$Label)
    $text = [string]$Value
    if ($text -cnotmatch '^[1-9][0-9]*$') {
        throw "$Label must be a positive decimal ID. Publication pins may be missing."
    }
    $number = 0L
    if (-not [long]::TryParse($text, [ref]$number)) {
        throw "$Label is outside the supported integer range."
    }
}

function Assert-Hash {
    param([object]$Value, [string]$Label)
    if ([string]$Value -cnotmatch '^[0-9a-fA-F]{64}$') {
        throw "$Label must be a SHA-256 digest. Publication pins may be missing."
    }
}

function Assert-RegularFile {
    param([string]$Path)
    $item = Get-Item -LiteralPath $Path -Force
    if ($item.PSIsContainer -or (($item.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0)) {
        throw "Expected a regular file, without a symbolic link: $Path"
    }
}

function Test-VerifiedFile {
    param([string]$Path, [string]$Sha256, [long]$Bytes = -1)
    if (-not (Test-Path -LiteralPath $Path)) {
        return $false
    }
    Assert-RegularFile -Path $Path
    if (($Bytes -ge 0) -and ((Get-Item -LiteralPath $Path -Force).Length -ne $Bytes)) {
        throw "Existing file has the wrong size; it will not be overwritten: $Path"
    }
    $actual = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash
    if ($actual -ine $Sha256) {
        throw "Existing file has the wrong SHA-256; it will not be overwritten: $Path"
    }
    return $true
}

function Get-VerifiedAsset {
    param([string]$AssetId, [string]$Name, [string]$Sha256, [long]$Bytes = -1)
    $target = Join-Path $script:DestinationPath $Name
    if (Test-VerifiedFile -Path $target -Sha256 $Sha256 -Bytes $Bytes) {
        Write-Host "Verified existing file: $Name"
        return
    }
    if ($Source -eq 'Local') {
        throw "Required file is missing from the local copy: $target"
    }

    # Windows PowerShell's native-command redirection can corrupt binary output.
    # Read the process's raw stdout stream directly into a new binary file.
    $temporary = Join-Path $script:DestinationPath ('.download-' + [guid]::NewGuid().ToString('N') + '.partial')
    $process = $null
    $stream = $null
    try {
        $startInfo = New-Object System.Diagnostics.ProcessStartInfo
        $startInfo.FileName = $script:GhPath
        # All variable arguments here are a fixed repository and validated decimal ID.
        $startInfo.Arguments = 'api --hostname github.com repos/' + $Repository + '/releases/assets/' + $AssetId + ' --header "Accept: application/octet-stream"'
        $startInfo.UseShellExecute = $false
        $startInfo.CreateNoWindow = $true
        $startInfo.RedirectStandardOutput = $true
        $startInfo.RedirectStandardError = $true
        $process = New-Object System.Diagnostics.Process
        $process.StartInfo = $startInfo
        $stream = [System.IO.File]::Open($temporary, [System.IO.FileMode]::CreateNew, [System.IO.FileAccess]::Write, [System.IO.FileShare]::None)
        if (-not $process.Start()) {
            throw "Could not start GitHub CLI for asset $AssetId."
        }
        # Drain stderr concurrently so neither pipe can block the download.
        $errorReader = $process.StandardError.ReadToEndAsync()
        $process.StandardOutput.BaseStream.CopyTo($stream)
        $stream.Flush()
        $stream.Dispose()
        $stream = $null
        $process.WaitForExit()
        $null = $errorReader.Result
        if ($process.ExitCode -ne 0) {
            throw "GitHub CLI failed for asset $AssetId (exit $($process.ExitCode)). Check gh authentication and release access; no existing file was overwritten."
        }
        if (-not (Test-VerifiedFile -Path $temporary -Sha256 $Sha256 -Bytes $Bytes)) {
            throw "Download did not produce asset $AssetId."
        }
        # File.Move fails if another file already occupies the destination.
        [System.IO.File]::Move($temporary, $target)
        Write-Host "Downloaded and verified: $Name"
    }
    finally {
        if ($null -ne $stream) { $stream.Dispose() }
        if ($null -ne $process) {
            try {
                if (-not $process.HasExited) { $process.Kill() }
            }
            catch { }
            $process.Dispose()
        }
        if (Test-Path -LiteralPath $temporary) {
            Remove-Item -LiteralPath $temporary -Force -ErrorAction SilentlyContinue
        }
    }
}

Assert-DecimalId -Value $CatalogueAssetId -Label 'Catalogue asset ID'
Assert-DecimalId -Value $ExpectedReleaseId -Label 'Release ID'
Assert-Hash -Value $CatalogueSha256 -Label 'Catalogue hash'

if ($Source -eq 'GitHub') {
    $ghCommand = Get-Command gh.exe -CommandType Application -ErrorAction Stop
    $script:GhPath = $ghCommand.Source
}
else {
    $script:GhPath = $null
}

if (-not (Test-Path -LiteralPath $Destination)) {
    if ($Source -eq 'Local') {
        throw "Local copy directory does not exist: $Destination"
    }
    $null = New-Item -ItemType Directory -Path $Destination
}
$destinationItem = Get-Item -LiteralPath $Destination -Force
if (-not $destinationItem.PSIsContainer) {
    throw "Destination is not a directory: $Destination"
}
$script:DestinationPath = $destinationItem.FullName

Get-VerifiedAsset -AssetId $CatalogueAssetId -Name $CatalogueName -Sha256 $CatalogueSha256
$cataloguePath = Join-Path $script:DestinationPath $CatalogueName
$utf8 = New-Object -TypeName System.Text.UTF8Encoding -ArgumentList @($false, $true)
$catalogue = [System.IO.File]::ReadAllText($cataloguePath, $utf8) | ConvertFrom-Json
foreach ($field in @('schema', 'repository', 'release_id', 'tag', 'snapshot_commit', 'assets')) {
    if ($catalogue.PSObject.Properties.Name -notcontains $field) {
        throw "Catalogue is missing its $field field."
    }
}
if ($catalogue.repository -cne $Repository) { throw 'Catalogue repository does not match the pinned repository.' }
Assert-DecimalId -Value $catalogue.release_id -Label 'Catalogue release ID'
if ([string]$catalogue.release_id -cne $ExpectedReleaseId) { throw 'Catalogue release does not match the pinned release.' }
if ([string]::IsNullOrWhiteSpace([string]$catalogue.schema) -or [string]::IsNullOrWhiteSpace([string]$catalogue.tag)) {
    throw 'Catalogue schema and tag must be present.'
}
if ([string]$catalogue.snapshot_commit -cnotmatch '^[0-9a-fA-F]{40}$') { throw 'Catalogue snapshot commit is invalid.' }
if (-not ($catalogue.assets -is [System.Array])) { throw 'Catalogue assets must be a JSON array.' }
$assets = @($catalogue.assets)
if (($assets.Count -eq 0) -or ($null -eq $assets[0])) { throw 'Catalogue contains no assets.' }
$seenNames = @{}
$seenIds = @{}
foreach ($asset in $assets) {
    foreach ($field in @('asset_id', 'name', 'bytes', 'sha256')) {
        if ($asset.PSObject.Properties.Name -notcontains $field) { throw "An asset is missing its $field field." }
    }
    Assert-DecimalId -Value $asset.asset_id -Label 'Asset ID'
    Assert-Hash -Value $asset.sha256 -Label 'Asset hash'
    $name = [string]$asset.name
    # A portable leaf name: no traversal, drive, stream, or reserved device name.
    if (($name -cnotmatch '^[A-Za-z0-9][A-Za-z0-9._-]*$') -or $name.EndsWith('.') -or
        ($name -imatch '^(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\.|$)')) {
        throw 'Catalogue contains an unsafe Windows file name.'
    }
    if (($name -ieq $CatalogueName) -or ([string]$asset.asset_id -ceq $CatalogueAssetId)) {
        throw 'Catalogue must not list itself as an asset.'
    }
    if ($seenNames.ContainsKey($name) -or $seenIds.ContainsKey([string]$asset.asset_id)) {
        throw 'Catalogue contains a duplicate file name or asset ID.'
    }
    $seenNames[$name] = $true
    $seenIds[[string]$asset.asset_id] = $true
    $byteCount = 0L
    if (([string]$asset.bytes -cnotmatch '^(0|[1-9][0-9]*)$') -or
        (-not [long]::TryParse([string]$asset.bytes, [ref]$byteCount))) {
        throw "Catalogue byte count is invalid for $name."
    }
    # Check every existing asset before starting any asset downloads.
    $null = Test-VerifiedFile -Path (Join-Path $script:DestinationPath $name) -Sha256 ([string]$asset.sha256) -Bytes $byteCount
}

foreach ($asset in $assets) {
    Get-VerifiedAsset -AssetId ([string]$asset.asset_id) -Name ([string]$asset.name) -Sha256 ([string]$asset.sha256) -Bytes ([long]$asset.bytes)
}

Write-Output ''
Write-Output ("Catalogue: {0}; asset ID: {1}; SHA-256: {2}" -f $CatalogueName, $CatalogueAssetId, $CatalogueSha256)
Write-Output ("Repository: {0}; release ID: {1}; tag: {2}; snapshot: {3}" -f $Repository, $ExpectedReleaseId, $catalogue.tag, $catalogue.snapshot_commit)
# Tab-separated output preserves full hashes even in a narrow terminal window.
Write-Output "name`tasset_id`tbytes`tsha256"
foreach ($asset in $assets) {
    Write-Output ("{0}`t{1}`t{2}`t{3}" -f $asset.name, $asset.asset_id, $asset.bytes, $asset.sha256)
}
Write-Output ("DOWNLOAD_AND_HASH_VERIFIED: {0} assets plus catalogue in {1}" -f $assets.Count, $script:DestinationPath)

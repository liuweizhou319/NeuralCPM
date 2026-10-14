$Obj = '79279_baseline.obj'
$Cross = '79279_baseline_framefield.txt'

$inputDir = Join-Path $PSScriptRoot 'input'
$exe = Join-Path $PSScriptRoot 'build\windows\x64\release\surfaceframefield_igl.exe'
$objPath = Join-Path $inputDir $Obj
$crossPath = Join-Path $inputDir $Cross

if (!(Test-Path -LiteralPath $exe -PathType Leaf)) { throw "Executable not found: $exe" }
if (!(Test-Path -LiteralPath $objPath -PathType Leaf)) { throw "OBJ not found: $objPath" }
if (!(Test-Path -LiteralPath $crossPath -PathType Leaf)) { throw "Cross field not found: $crossPath" }

& $exe $objPath $crossPath
exit $LASTEXITCODE

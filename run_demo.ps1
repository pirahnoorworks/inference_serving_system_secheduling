$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
python .\demo_multipipeline.py --jobs 400 --seed 7 --plot

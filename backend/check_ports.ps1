$ports = @(8000,3000)
foreach ($p in $ports) {
    Write-Output "--- Port $p ---"
    netstat -ano | findstr ":$p"
}

Write-Output "\nProcesses matching uvicorn/node/react-scripts:"
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'uvicorn' -or $_.CommandLine -match 'node' -or $_.CommandLine -match 'react-scripts' } | Select-Object ProcessId, CommandLine | Format-List

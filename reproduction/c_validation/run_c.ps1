param(
    [ValidateSet('baseline', 'original-check', 'full-check', 'extension', 'probe', 'plot', 'diagnose', 'combined-prepare', 'combined-check', 'precision-test', 'precision-prepare', 'precision-check', 'precision-export', 'precision-portable')]
    [string]$Step = 'diagnose'
)

# 只设置本次进程的环境；不改注册表和系统 PATH。
$ErrorActionPreference = 'Stop'
$taskRepo = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$taskPython = Join-Path $taskRepo '.venv-c310\Scripts\python.exe'
$taskSpark = 'C:\Users\33925\Documents\Codex\work\kc_spark'
$taskJava = 'C:\Users\33925\Documents\Codex\work\zulu17.70.201-ca-jre17.0.21.1-win_x64'
$env:JAVA_HOME = $taskJava
$env:HADOOP_HOME = 'C:\Users\33925\Documents\Codex\work\kc_hadoop'
$env:SPARK_HOME = $taskSpark
$env:PYSPARK_PYTHON = $taskPython
$env:PYSPARK_DRIVER_PYTHON = $taskPython
$env:PYTHONPATH = (Join-Path $PSScriptRoot 'runtime_compat') + ';' + (Join-Path $taskRepo 'src') + ';' + (Join-Path $taskSpark 'python')
if ($Step -eq 'plot') { $env:PYTHONPATH += ';C:\Users\33925\.codex\skills\nature-figure\scripts' }
$env:SPARK_LOCAL_IP = '127.0.0.1'
$env:CUDA_VISIBLE_DEVICES = '-1'
$env:KERAS_BACKEND = 'tensorflow'
$env:TF_CPP_MIN_LOG_LEVEL = '2'
$env:PYTHONUTF8 = '1'
$env:OMP_NUM_THREADS = '2'
$env:TF_NUM_INTRAOP_THREADS = '2'
$env:TF_NUM_INTEROP_THREADS = '2'
$env:TEMP = 'C:\Users\33925\Documents\Codex\work\kc_tmp'
$env:KAMAE_C_TEMP_ROOT = $env:TEMP
$env:KAMAE_C_REPO_ROOT = $taskRepo
$env:PYSPARK_SUBMIT_ARGS = '--conf spark.hadoop.fs.file.impl=cvalidation.InheritedAclLocalFileSystem pyspark-shell'
$env:TMP = $env:TEMP
$env:MPLCONFIGDIR = Join-Path $PSScriptRoot '.cache\mpl'
$env:SPARK_LOCAL_DIRS = Join-Path $env:TEMP 'spark'
New-Item -ItemType Directory -Force -Path $env:TEMP,$env:MPLCONFIGDIR,$env:SPARK_LOCAL_DIRS | Out-Null
$env:PATH = (Join-Path $env:HADOOP_HOME 'bin') + ';' + (Join-Path $taskJava 'bin') + ';' + $env:PATH
$taskResults = Join-Path $PSScriptRoot 'results'
New-Item -ItemType Directory -Force -Path $taskResults | Out-Null
$taskLog = Join-Path $taskResults ($Step + '.log')
Push-Location $taskRepo
try {
    switch ($Step) {
        'baseline' { $taskScript = Join-Path $taskRepo 'reproduction\scripts\run_movielens_pipeline.py' }
        'original-check' { $taskScript = Join-Path $taskRepo 'reproduction\scripts\verify_movielens_parity.py' }
        'full-check' { $taskScript = Join-Path $PSScriptRoot 'validate_full.py' }
        'extension' { $taskScript = Join-Path $PSScriptRoot 'extension_bucketize.py' }
        'probe' { $taskScript = Join-Path $PSScriptRoot 'probe_bucketize.py' }
        'plot' { $taskScript = Join-Path $PSScriptRoot 'plot_extension.py' }
        'diagnose' { $taskScript = Join-Path $PSScriptRoot 'diagnose.py' }
        'combined-prepare' { $taskScript = Join-Path $PSScriptRoot 'combined_extension.py'; $env:KAMAE_COMBINED_PHASE = 'prepare' }
        'combined-check' { $taskScript = Join-Path $PSScriptRoot 'combined_extension.py'; $env:KAMAE_COMBINED_PHASE = 'compare' }
        'precision-test' { $taskScript = Join-Path $PSScriptRoot 'precision_experiment.py'; $env:KAMAE_PRECISION_PHASE = 'test' }
        'precision-prepare' { $taskScript = Join-Path $PSScriptRoot 'precision_experiment.py'; $env:KAMAE_PRECISION_PHASE = 'prepare' }
        'precision-check' { $taskScript = Join-Path $PSScriptRoot 'precision_experiment.py'; $env:KAMAE_PRECISION_PHASE = 'compare' }
        'precision-export' { $taskScript = Join-Path $PSScriptRoot 'precision_experiment.py'; $env:KAMAE_PRECISION_PHASE = 'export' }
        'precision-portable' { $taskScript = Join-Path $PSScriptRoot 'precision_portable_check.py' }
    }
    $taskStarted = Get-Date -Format o
    & $taskPython -u $taskScript *> $taskLog
    $taskCode = $LASTEXITCODE
    [ordered]@{step=$Step; start=$taskStarted; end=(Get-Date -Format o); exit_code=$taskCode; script=$taskScript} |
        ConvertTo-Json | Set-Content -LiteralPath (Join-Path $taskResults ($Step + '_run.json')) -Encoding utf8
    Get-Content -LiteralPath $taskLog -Tail 35
    exit $taskCode
} finally {
    Pop-Location
}

@echo off
setlocal enabledelayedexpansion
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONDONTWRITEBYTECODE=1
set KMP_DUPLICATE_LIB_OK=TRUE

echo ========================================================================
echo   SUITE DE COMPARACION: MODELOS BIFUSION vs MAMBALITE3D (HECKTOR)
echo ========================================================================

:: Activar entorno conda mambalite3d si existe
if exist "%USERPROFILE%\miniconda3\Scripts\activate.bat" (
    call "%USERPROFILE%\miniconda3\Scripts\activate.bat" mambalite3d
) else if exist "%USERPROFILE%\anaconda3\Scripts\activate.bat" (
    call "%USERPROFILE%\anaconda3\Scripts\activate.bat" mambalite3d
)

:: Paso 1: Ejecutar analisis comparativo exhaustivo (metricas, estadistica, Pareto, radar)
echo.
echo [1/3] Ejecutando comparativa analitica y estadistica pareada...
python scripts\comparar_bifusion_mamba.py %*
if errorlevel 1 (
    echo [ERROR] Fallo al ejecutar scripts\comparar_bifusion_mamba.py
    pause
    exit /b 1
)

:: Paso 2: Ejecutar comparativa de curvas de entrenamiento y convergencia
echo.
echo [2/3] Generando comparativa de dinamica de entrenamiento y perdida...
python scripts\comparar_curvas_entrenamiento.py
if errorlevel 1 (
    echo [AVISO] Fallo o no se encontraron curvas en scripts\comparar_curvas_entrenamiento.py
)

:: Paso 3: Resumen de archivos generados
echo.
echo ========================================================================
echo   ARCHIVOS Y REPORTES GENERADOS SATISFACTORIAMENTE:
echo ========================================================================
echo   - Reporte Markdown:  estadisticas\comparativa_bifusion_mamba.md
echo   - Resumen en CSV:    estadisticas\comparativa_bifusion_mamba.csv
echo   - Datos en JSON:     estadisticas\comparativa_bifusion_mamba.json
echo   - Graficas generadas en estadisticas\graficas\:
echo       * comparativa_metricas_barras.png
echo       * comparativa_distribucion_dice.png
echo       * comparativa_pareto_eficiencia.png
echo       * comparativa_radar.png
echo       * comparativa_curvas_entrenamiento.png
echo ========================================================================
echo   Para comparar cortes 3D de un paciente especifico:
echo   python scripts\visualizar_comparativa_segmentacion.py --case CHUM-001
echo ========================================================================
echo.
exit /b 0

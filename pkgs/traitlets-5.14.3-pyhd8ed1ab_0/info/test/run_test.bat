



pip check
IF %ERRORLEVEL% NEQ 0 exit /B 1
coverage run --source=traitlets --parallel --branch -m pytest -vv
IF %ERRORLEVEL% NEQ 0 exit /B 1
coverage run --source=traitlets --parallel --branch -m pytest -vv --pyargs traitlets
IF %ERRORLEVEL% NEQ 0 exit /B 1
coverage combine
IF %ERRORLEVEL% NEQ 0 exit /B 1
coverage report --show-missing --skip-covered --fail-under=80
IF %ERRORLEVEL% NEQ 0 exit /B 1
mypy -p traitlets
IF %ERRORLEVEL% NEQ 0 exit /B 1
exit /B 0

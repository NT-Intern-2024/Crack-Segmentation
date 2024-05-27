@ECHO OFF

FOR %%a IN (*.jpg) DO (
  SET newname=%%~na_.jpg
  REN "%%a" "%%newname%%"
  ECHO Renaming "%%a" to "%%newname%%"
)

ECHO Finished renaming files.

PAUSE
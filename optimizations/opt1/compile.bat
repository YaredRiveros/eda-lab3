@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat"
cl /EHsc /std:c++17 /O2 /Fe:trajectory_search_opt1.exe main.cpp

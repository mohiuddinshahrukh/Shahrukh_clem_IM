@echo off
REM Run function_detective for many ChatGPT/OpenAI models + your other models
REM Continues even if one run fails

echo Starting all clem runs for function_detective...

REM -------- OpenAI / ChatGPT models --------
call clem run -g function_detective -m gpt-4o-2024-08-06  || echo failed
call clem run -g function_detective -m gpt-4o-2024-05-13  || echo failed
call clem run -g function_detective -m gpt-4o-mini-2024-07-18  || echo failed

call clem run -g function_detective -m gpt-4.1-2025-04-14  || echo failed
call clem run -g function_detective -m gpt-4.1-mini  || echo failed
call clem run -g function_detective -m gpt-4.1-nano  || echo failed

call clem run -g function_detective -m gpt-4-turbo-2024-04-09  || echo failed
call clem run -g function_detective -m gpt-4-0125-preview || echo failed
call clem run -g function_detective -m gpt-4-1106-preview  || echo failed
call clem run -g function_detective -m gpt-4-0613  || echo failed
call clem run -g function_detective -m gpt-4-0314  || echo failed

call clem run -g function_detective -m gpt-3.5-turbo  || echo failed
call clem run -g function_detective -m gpt-3.5-turbo-0125  || echo failed
call clem run -g function_detective -m gpt-3.5-turbo-1106  || echo failed
call clem run -g function_detective -m gpt-3.5-turbo-0613  || echo failed

echo.
echo All runs attempted. Now transcribing, scoring, and evaluating (per results folder)...

REM Process each results_* folder so you don’t rely on “latest/default” behavior
for /d %%D in (results_*) do (
  echo === Processing %%D ===
  call clem transcribe -r %%D || echo transcribe failed for %%D
  call clem score -r %%D || echo score failed for %%D
  call clem eval -r %%D || echo eval failed for %%D
)

echo Done!
pause
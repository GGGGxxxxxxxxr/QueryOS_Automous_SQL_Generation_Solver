import subprocess
import json
import logging
import snowflake.connector
from typing import Dict, Any

logger = logging.getLogger(__name__)

import os
import re
import shlex
import subprocess
from typing import Dict, Any, Optional, Tuple



class Spider2Tools:
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.timeout = self.config.get("timeout", 120)
        self.current_dir = None

        # JSON-only paging controls
        self.file_view_line_threshold = self.config.get("file_view_line_threshold", 500)
        self.file_view_chunk_lines = self.config.get("file_view_chunk_lines", 500)

        # For JSON reads, cap output by lines (not chars)
        self.max_json_output_chars = self.config.get("max_json_output_chars", 20000)  # safety only

    def execute_cmd(self, command: str) -> str:
        try:
            if command.strip().startswith("cd "):
                return self._handle_cd_command(command)

            # ✅ JSON-only paging: avoid char truncation + avoid middle truncation
            paged = self._maybe_paginate_json_view(command)
            if paged is not None:
                return f"CMD_output: {paged}"

            # Normal execution (non-JSON viewing commands)
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=self.current_dir,
            )
            output = (result.stdout or "") + (result.stderr or "")
            return f"CMD_output: {output}" if output.strip() else f"Exit code: {result.returncode}"

        except subprocess.TimeoutExpired:
            return f"Error: Command timed out after {self.timeout} seconds"
        except Exception as e:
            return f"Error: {str(e)}"

    # ------------------------------------------------------------------
    # JSON-only file viewing: line-based, head-only, NO middle truncation
    # ------------------------------------------------------------------
    def _maybe_paginate_json_view(self, command: str) -> Optional[str]:
        """
        Intercept simple JSON viewing commands (cat/head/tail/sed on *.json).
        If file is large (lines > threshold), return ONLY a head window (or requested window),
        plus a structured warning with next_cmd. Never stitch head+tail. Never truncate middle.
        """
        cmd = command.strip()

        # Keep it simple: do not interfere with pipelines / redirects / compound commands
        if any(tok in cmd for tok in ["|", ">", "<", "&&", ";"]):
            return None

        parsed = self._parse_file_view_command(cmd)
        if parsed is None:
            return None

        view_kind, file_path, requested_window = parsed
        abs_path = self._resolve_path(file_path)
        if not abs_path or not os.path.isfile(abs_path):
            return None

        # JSON-only
        if not abs_path.lower().endswith(".json"):
            return None

        total_lines = self._wc_l(abs_path)
        if total_lines is None:
            return None

        # Small JSON: just run original command normally
        if total_lines <= self.file_view_line_threshold:
            return None

        chunk = self.file_view_chunk_lines

        # Determine window to show
        # Default: HEAD chunk (1..chunk). If user asked a window, honor it but cap to chunk.
        if requested_window is None:
            start, end = 1, min(chunk, total_lines)
        else:
            start, end = requested_window

            # Handle tail sentinel: (-n, -1) means last n lines
            if start < 0 and end == -1:
                n = abs(start)
                end = total_lines
                start = max(1, total_lines - n + 1)

            start = max(1, start)
            end = min(end, total_lines)

            # Cap to chunk length
            if end - start + 1 > chunk:
                end = start + chunk - 1

        content = self._read_lines_window(abs_path, start, end)

        # Safety: cap chars if content is unexpectedly huge (shouldn't happen with line window)
        if self.max_json_output_chars and len(content) > self.max_json_output_chars:
            # Still NO middle truncation: keep head only.
            content = content[: self.max_json_output_chars]

        # Warning: if we are showing from start=1, we can say shown_lines=1-end.
        # If not starting at 1, we still provide exact shown_lines and next_cmd.
        next_start = end + 1
        next_cmd = None
        if next_start <= total_lines:
            next_cmd = f"tail -n +{next_start} '{abs_path}' | head -n {chunk}"

        warn_lines = [
            "[FILE_TOO_LARGE_WARNING]",
            f"file={abs_path}",
            f"total_lines={total_lines}",
            f"shown_lines={start}-{end}",
            f"chunk_lines={chunk}",
            "note=Large JSON file. Only a small window is shown.",
            "hint=To inspect later parts, read another window of lines (e.g., `sed -n '100,200p' <file>`).",
            "[/FILE_TOO_LARGE_WARNING]",
        ]

        warn = "\n".join(warn_lines)
        return warn + "\n" + content

        '''
        if next_cmd:
            warn_lines.append(f"next_cmd={next_cmd}")
        warn_lines += [
            "refine_cmds:",
            f"- head: head -n {chunk} '{abs_path}'",
            f"- next_chunk: {next_cmd}" if next_cmd else "- next_chunk: (none)",
            f"- window_sed: sed -n '{next_start},{min(next_start+chunk-1,total_lines)}p' '{abs_path}'" if next_cmd else "- window_sed: (none)",
            f"- search: grep -n \"<keyword>\" '{abs_path}' | head",
            "[/FILE_TOO_LARGE_WARNING]",
        ]
        '''
        warn = "\n".join(warn_lines)
        
        return warn + "\n" + content

    def _parse_file_view_command(self, cmd: str) -> Optional[Tuple[str, str, Optional[Tuple[int, int]]]]:
        """
        Supports:
          - cat <file.json>
          - head -n N <file.json>     => window (1..N)
          - tail -n N <file.json>     => sentinel (-N,-1) meaning last N lines
          - sed -n 'A,Bp' <file.json> => window (A..B)
        """
        try:
            parts = shlex.split(cmd)
        except Exception:
            return None
        if not parts:
            return None

        tool = parts[0]
        if tool == "cat" and len(parts) == 2:
            return ("cat", parts[1], None)

        if tool == "head":
            if len(parts) >= 4 and parts[1] == "-n":
                try:
                    n = int(parts[2])
                except Exception:
                    return None
                return ("head", parts[3], (1, n))
            return None

        if tool == "tail":
            if len(parts) >= 4 and parts[1] == "-n":
                try:
                    n = int(parts[2])
                except Exception:
                    return None
                return ("tail", parts[3], (-n, -1))
            return None

        if tool == "sed":
            if len(parts) >= 4 and parts[1] == "-n":
                m = re.match(r"^(\d+),(\d+)p$", parts[2].strip("'\""))
                if m:
                    a = int(m.group(1))
                    b = int(m.group(2))
                    return ("sed", parts[3], (a, b))
            return None

        return None

    def _read_lines_window(self, path: str, start: int, end: int) -> str:
        cmd = f"sed -n '{start},{end}p' '{path}'"
        r = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=self.timeout,
            cwd=self.current_dir,
        )
        return (r.stdout or "") + (r.stderr or "")

    def _wc_l(self, path: str) -> Optional[int]:
        try:
            r = subprocess.run(
                f"wc -l '{path}'",
                shell=True,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=self.current_dir,
            )
            txt = (r.stdout or "").strip()
            if not txt:
                return None
            return int(txt.split()[0])
        except Exception:
            return None

    def _resolve_path(self, p: str) -> Optional[str]:
        try:
            if not p:
                return None
            p = os.path.expanduser(p)
            if os.path.isabs(p):
                return os.path.normpath(p)
            base = self.current_dir or os.getcwd()
            return os.path.normpath(os.path.join(base, p))
        except Exception:
            return None

    def _handle_cd_command(self, command: str) -> str:
        parts = command.strip().split(maxsplit=1)
        target_dir = os.path.expanduser("~") if len(parts) < 2 else parts[1]
        try:
            if not os.path.isabs(target_dir):
                if self.current_dir:
                    target_dir = os.path.join(self.current_dir, target_dir)
                else:
                    target_dir = os.path.abspath(target_dir)
            target_dir = os.path.normpath(target_dir)
            if os.path.isdir(target_dir):
                self.current_dir = target_dir
                return f"CMD_output: Changed directory to {self.current_dir}"
            else:
                return f"CMD_output: cd: {target_dir}: No such file or directory"
        except Exception as e:
            return f"CMD_output: cd: {str(e)}"
    
    def execute_snowflake_sql(self, sql: str) -> str:
        """执行Snowflake SQL"""
        try:
            credentials = self._get_snowflake_credentials()
            
            conn = snowflake.connector.connect(**credentials)
            cursor = conn.cursor()
            cursor.execute(f"ALTER SESSION SET STATEMENT_TIMEOUT_IN_SECONDS = {self.timeout}")
            cursor.execute(sql)
            
            if cursor.description:
                headers = [desc[0] for desc in cursor.description]
                rows = cursor.fetchall()
                
                if rows:
                    # 简单格式化结果
                    result_lines = ["\t".join(headers)]
                    for row in rows[:100]:  # 限制行数
                        result_lines.append("\t".join(str(cell) for cell in row))
                    
                    result = "\n".join(result_lines)
                    if len(rows) > 100:
                        result += f"\n... ({len(rows)} total rows, showing first 100)"
                    
                    # 限制输出总长度
                    if len(result) > self.max_output_length:
                        result = result[:self.max_output_length] + "\n... (output truncated)"
                    
                    return f"SQL_output: {result}"
                else:
                    return "SQL_output: No rows returned"
            else:
                conn.commit()
                return "SQL_output: Query executed successfully"
                
        except Exception as e:
            return f"SQL_output: Error: {str(e)}"
        finally:
            if 'conn' in locals():
                conn.close()
    
    def _get_snowflake_credentials(self) -> Dict[str, str]:
        """获取Snowflake凭证"""
        credentials_path = "/efs/sunkexua/src/verl/data/spider2-snow/credentials/snowflake_credential.json"
        try:
            with open(credentials_path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading Snowflake credentials: {e}")
            raise
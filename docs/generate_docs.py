#!/usr/bin/env python3

# Copyright 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import argparse
import logging
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from enum import Enum
from logging.handlers import RotatingFileHandler
from typing import Dict, List, Optional, Tuple, Union

from github_slugger import GithubSlugger

# Global constants
server_abspath = os.environ.get("SERVER_ABSPATH", os.getcwd())
server_docs_abspath = os.path.join(server_abspath, "docs")

"""
TODO: Needs to handle cross-branch linkage.

For example, server/docs/user_guide/architecture.md on branch 24.12 links to
server/docs/user_guide/model_analyzer.md on main branch. In this case, the
hyperlink of model_analyzer.md should be a URL instead of relative path.

Another example can be server/docs/user_guide/model_analyzer.md on branch 24.12
links to a file in server repo with relative path. Currently all URLs are
hardcoded to main branch. We need to make sure that the URL actually points to the
correct branch. We also need to handle cases like deprecated or removed files from
older branch to avoid 404 error code.
"""
# Regex patterns
http_patn = r"^https?://"
http_reg = re.compile(http_patn)
tag_patn = "/(?:blob|tree)/main"
triton_repo_patn = rf"{http_patn}github.com/triton-inference-server"
triton_github_url_reg = re.compile(
    rf"{triton_repo_patn}/([^/#]+)(?:{tag_patn})?/*([^#]*)\s*(?=#|$)"
)
# Hyperlink in a .md file, excluding embedded images.
hyperlink_reg = re.compile(r"((?<!\!)\[[^\]]+\]\(\s*)([^)]+?)(\s*\))")

# Load exclusion patterns
with open(f"{server_docs_abspath}/exclusions.txt") as f:
    exclude_patterns = f.read().strip().split("\n")

white_list_domains = [
    "platform.openai.com",
    "docs.openvinotoolkit.org",
    "www.tensorflow.org",
    "en.wikipedia.org",
    "www.adept.ai",
    "huggingface.co",
    "nbviewer.org",
    "mailto:psirt@nvidia.com",
]


class LinkType(Enum):
    URL = 0
    RELATIVE_PATH = 1


def type_of_link(link: str) -> LinkType:
    """
    Determine if a link is a URL or a relative path.

    Args:
        link: The link string to check

    Returns:
        LinkType.URL if the link starts with http:// or https://,
        LinkType.RELATIVE_PATH otherwise
    """
    if http_reg.match(link):
        return LinkType.URL
    else:
        return LinkType.RELATIVE_PATH


# Setup logger once
def setup_logger(
    name: str,
    log_file: str,
    level: int = logging.INFO,
    max_bytes: int = 1048576,
    backup_count: int = 5,
) -> logging.Logger:
    """
    Set up a logger with both file and console handlers.

    Creates a rotating file handler and a console handler with colored output.
    Prevents duplicate handlers if called multiple times.

    Args:
        name: Logger name (typically the script name)
        log_file: Path to the log file
        level: Logging level (default: INFO)
        max_bytes: Maximum size of log file before rotation (default: 1MB)
        backup_count: Number of backup log files to keep (default: 5)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent adding multiple handlers if the function is called multiple times
    if not logger.handlers:
        # Create handlers
        file_handler = RotatingFileHandler(
            log_file, maxBytes=max_bytes, backupCount=backup_count
        )
        console_handler = logging.StreamHandler()

        # Set the logging level for handlers
        file_handler.setLevel(level)
        console_handler.setLevel(level)

        # Create a logging format
        BLUE = "\033[94m"
        RESET = "\033[0m"
        formatter = logging.Formatter(
            f"{BLUE}%(asctime)s - %(name)s - %(levelname)s - {RESET}%(message)s"
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add handlers to the logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    return logger


parser = argparse.ArgumentParser(description="Setup Triton Server Docs")
parser.add_argument(
    "--repo-tag",
    type=str,
    default=os.environ.get("TRITON_SERVER_REPO_TAG", "main"),
    help="Repository tags in format value",
)
parser.add_argument(
    "--log-file",
    type=str,
    default=os.environ.get("TRITON_SERVER_DOCS_LOG_FILE", "/tmp/docs.log"),
    help="The path to the log file",
)
parser.add_argument(
    "--repo-file",
    default="repositories.txt",
    help="File which lists the repositories to add. File should be"
    " one repository name per line, newline separated.",
)
parser.add_argument(
    "--github-organization",
    type=str,
    default=os.environ.get(
        "TRITON_SERVER_REPO_ORG", "https://github.com/triton-inference-server"
    ),
    help="GitHub organization name",
)
args = parser.parse_args()


logger = setup_logger(os.path.basename(__file__), args.log_file, level=logging.DEBUG)
logger.info(f"Defined arguments: {args}")


def run_command(command: str) -> None:
    """
    Execute a shell command using subprocess and log the execution.

    Args:
        command: Shell command string to execute

    Raises:
        subprocess.CalledProcessError: If the command returns a non-zero exit code
    """
    logger.info(f"Running command: {command}")
    subprocess.run(
        command,
        shell=True,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def clone_from_github(repo: str, tag: str, org: str) -> None:
    """
    Clone a repository from GitHub using git.

    Clones a specific branch/tag of a repository. For model_navigator repo,
    uses a hardcoded branch name. Uses single-branch cloning for efficiency.

    Args:
        repo: Repository name (e.g., "server", "model_analyzer")
        tag: Branch or tag name to clone (e.g., "main", "24.12")
        org: GitHub organization URL (e.g., "https://github.com/triton-inference-server")

    Raises:
        subprocess.CalledProcessError: If git clone fails
    """
    logger.info(f"Cloning... {org}/{repo}.git@{tag}")
    repo_url = f"{org}/{repo}.git"

    if tag:
        if re.match("model_navigator", repo):
            tag = "yinggeh/tri-318-validate-links-and-fix-broken-ones-in-docs"

        clone_command = ["git", "clone", "--branch", tag, "--single-branch", repo_url]
    else:
        clone_command = ["git", "clone", repo_url]

    subprocess.run(clone_command, check=True)
    logger.info(f"Successfully cloned... {org}/{repo}.git@{tag}")


def is_excluded(file_path: str) -> bool:
    """
    Check if a file path should be excluded from processing.

    Compares the file path against exclusion patterns loaded from exclusions.txt.
    A file is excluded if its path is under any exclusion pattern path.

    Args:
        file_path: Path to the file to check

    Returns:
        True if the file should be excluded, False otherwise
    """
    file_abspath = os.path.abspath(file_path)
    for pattern in exclude_patterns:
        exclude_abspath = os.path.abspath(pattern)
        if os.path.commonpath([file_abspath, exclude_abspath]) == exclude_abspath:
            return True
    return False


def get_git_repo_name(file_path: str) -> str:
    """
    Determine the Git repository name for a given file path.

    Queries git to get the remote origin URL and extracts the repository name.

    Args:
        file_path: Path to a file within a git repository

    Returns:
        Repository name (e.g., "server", "model_analyzer")

    Raises:
        subprocess.CalledProcessError: If git command fails
    """
    directory = os.path.dirname(file_path)
    remote_url = (
        subprocess.check_output(["git", "-C", directory, "remote", "get-url", "origin"])
        .decode()
        .strip()
    )

    # Extract repository name from the remote URL
    if remote_url.endswith(".git"):
        remote_url = remote_url[:-4]
    return os.path.basename(remote_url)


def replace_url_with_relpath(url: str, src_doc_path: str) -> str:
    """
    Replace Triton Inference Server GitHub URLs with relative paths for:
    1. URL is a doc file (e.g., ".md" file).
    2. URL is a directory with README.md and ends with "#<section>".

    Examples:
        https://github.com/triton-inference-server/server/blob/main/docs/protocol#restricted-protocols
        https://github.com/triton-inference-server/server/blob/main/docs/protocol/extension_shared_memory.md
        https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md#dynamic-batcher
    """
    m = triton_github_url_reg.match(url)
    if not m:
        return url

    target_repo_name = m.group(1)
    logger.debug(f"Found target repository: {target_repo_name}")
    target_relpath_from_target_repo = os.path.normpath(m.groups("")[1])
    logger.debug(
        f"Found target relative path from target repository: {target_relpath_from_target_repo}"
    )
    section = url[len(m.group(0)) :]
    logger.debug(f"Found section: {section}")
    valid_hashtag = section not in ["", "#"] and section.startswith("#")

    target_path = (
        os.path.join(server_abspath, target_relpath_from_target_repo)
        if target_repo_name == "server"
        else os.path.join(
            server_docs_abspath, target_repo_name, target_relpath_from_target_repo
        )
    )
    logger.debug(f"Found target path: {target_path}")
    # Return URL if it points to a path outside server/docs
    if os.path.commonpath([server_docs_abspath, target_path]) != server_docs_abspath:
        return url
    logger.debug(
        f"Target path is under server/docs directory: {os.path.commonpath([server_docs_abspath, target_path]) == server_docs_abspath}"
    )
    # Check if target is valid for conversion
    is_md_file = _is_markdown_file(target_path)
    logger.debug(f"Target path is a valid .md file: {is_md_file}")
    is_dir_with_readme = (
        os.path.isdir(target_path)
        and os.path.isfile(os.path.join(target_path, "README.md"))
        and valid_hashtag
        and not is_excluded(os.path.join(target_path, "README.md"))
    )
    logger.debug(f"Target path is a directory with README.md: {is_dir_with_readme}")
    if is_dir_with_readme:
        target_path = os.path.join(target_path, "README.md")
    else:
        return url
    logger.debug(
        f"Target path is a valid .md file or a directory with README.md: {is_md_file or is_dir_with_readme}"
    )

    relpath = os.path.relpath(target_path, start=os.path.dirname(src_doc_path))
    logger.debug(f"Found relative path: {relpath}")
    return re.sub(triton_github_url_reg, relpath, url, 1)


def replace_relpath_with_url(relpath: str, src_doc_path: str) -> str:
    """
    Convert relative paths to GitHub URLs when appropriate.

    Replaces relative paths with Triton Inference Server GitHub URLs in the following cases:
    1. Relative path points to a non-markdown file (e.g., .pbtxt, .py)
    2. Relative path points to a directory without README.md or without a section anchor
    3. Relative path does not exist (would show 404)

    Keeps relative paths for valid markdown files within the docs directory.

    Args:
        relpath: Relative path from the source document
        src_doc_path: Absolute path to the source document file

    Returns:
        GitHub URL string if conversion is needed, original relative path otherwise

    Examples:
        ../examples/model_repository -> https://github.com/.../examples/model_repository
        ../examples/model_repository/inception_graphdef/config.pbtxt -> https://github.com/.../config.pbtxt
    """
    target_path = relpath.rsplit("#", 1)[0]
    section = relpath[len(target_path) :]
    valid_hashtag = section not in ["", "#"]

    if relpath.startswith("#"):
        target_path = os.path.basename(src_doc_path)

    target_path = os.path.normpath(
        os.path.join(os.path.dirname(src_doc_path), target_path)
    )
    src_git_repo_name = get_git_repo_name(src_doc_path)

    src_repo_abspath = (
        server_abspath
        if src_git_repo_name == "server"
        else os.path.join(server_docs_abspath, src_git_repo_name)
    )

    # Assert target path is under the current repo directory
    assert os.path.commonpath([src_repo_abspath, target_path]) == src_repo_abspath

    target_path_from_src_repo = os.path.relpath(target_path, start=src_repo_abspath)

    # For example, target_path of "../protocol#restricted-protocols" should be "<path-to-server>/server/docs/protocol/README.md"
    if (
        os.path.isdir(target_path)
        and valid_hashtag
        and os.path.isfile(os.path.join(target_path, "README.md"))
    ):
        relpath = os.path.join(relpath.rsplit("#", 1)[0], "README.md") + section
        target_path = os.path.join(target_path, "README.md")

    # Keep relpath if it's a valid .md file in docs
    if (
        _is_markdown_file(target_path)
        and os.path.commonpath([server_docs_abspath, target_path])
        == server_docs_abspath
    ):
        return relpath

    return f"https://github.com/triton-inference-server/{src_git_repo_name}/blob/main/{target_path_from_src_repo}{section}"


# Global cache for URL validation to minimize HTTP requests
# Format: {base_url: (exists: bool, content: str or None, error_message: str)}
_url_cache: Dict[str, Tuple[bool, Optional[str], str]] = {}

# Collect validation errors to report at the end
# Format: [(result_link, original_link, src_doc_path, error_message), ...]
_validation_errors: List[Tuple[str, str, str, str]] = []


def _markdown_heading_to_anchor(heading: str, slugger: GithubSlugger) -> str:
    """
    Convert a markdown heading to GitHub's anchor format.

    Example:
        Input:  "##### Using the [generate endpoint](https://github.com/triton-inference-server/server/blob/main/docs/protocol/extension_generate.md)"
        Output: "using-the-generate-endpoint"

    Strips markdown hyperlinks from the heading text before converting to anchor,
    since headings may contain inline links. Uses GithubSlugger to generate
    the anchor in the same format GitHub uses.

    Args:
        heading: Markdown heading text (may contain markdown links)
        slugger: GithubSlugger instance for anchor generation

    Returns:
        Anchor string matching GitHub's format
    """

    # Heading may contain hyperlink, e.g. ##### Using the [generate endpoint](https://github.com/triton-inference-server/server/blob/main/docs/protocol/extension_generate.md)
    # Remove all markdown hyperlinks by using hyperlink_reg to replace them with just the link text, keeping possible multiple links.
    # For example, "[foo](bar) [baz](qux)" -> "foo baz"
    def _strip_markdown_links(text):
        # Replace all markdown links [text](url) (excluding images) with just the link text.
        return hyperlink_reg.sub(lambda m: m.group(1)[1:-2], text)

    flat_heading = _strip_markdown_links(heading)

    expected_anchor = slugger.slug(flat_heading)
    return expected_anchor


def _is_file_url(url: str) -> bool:
    """
    Determine if a URL likely points to a file or a directory.

    Checks if the URL path has a file extension (contains a dot in the basename).
    This helps distinguish between file URLs and directory URLs.

    Args:
        url: URL to check

    Returns:
        True if URL appears to point to a file, False if it's likely a directory
    """
    path = url.split("?")[0].split("#")[0]
    basename = os.path.basename(path)
    return "." in basename


def _get_github_raw_url(url: str, anchor: Optional[str]) -> Optional[str]:
    """
    Convert GitHub blob/tree URL to raw.githubusercontent.com URL.

    Example:
        Input:  "https://github.com/triton-inference-server/server/blob/main/docs/README.md", anchor=None
        Output: "https://raw.githubusercontent.com/triton-inference-server/server/main/docs/README.md"

        Input:  "https://github.com/triton-inference-server/server", anchor="triton-inference-server"
        Output: "https://raw.githubusercontent.com/triton-inference-server/server/main/README.md"

    Converts URLs pointing to rendered GitHub pages to their corresponding raw file URLs
    for direct content access and anchor validation.

    Args:
        url: GitHub URL (blob or tree format)
        anchor: Optional anchor/fragment identifier

    Returns:
        Raw GitHub URL string, or None if conversion is not applicable
    """
    if not "github.com/" in url:
        return None

    # Case: https://github.com/triton-inference-server/server#triton-inference-server
    if len(url.split("github.com/")[-1].split("/")) == 2:
        url += "/tree/main"

    if "/blob/" in url and _is_file_url(url):
        return (
            url.replace("github.com", "raw.githubusercontent.com")
            .replace("/blob/", "/")
            .replace("/tree/", "/")
        )
    if anchor and not _is_file_url(url):
        return (
            url.replace("github.com", "raw.githubusercontent.com")
            .replace("/blob/", "/")
            .replace("/tree/", "/")
            + "/README.md"
        )
    return None


def _add_browser_headers(req: urllib.request.Request) -> urllib.request.Request:
    """
    Add browser-like HTTP headers to a request to avoid anti-bot blocking.

    Some websites block requests that don't have typical browser headers.
    This function adds User-Agent, Accept, Accept-Language, and other
    headers to make the request appear more browser-like.

    Args:
        req: urllib.request.Request object to modify

    Returns:
        Modified request object
    """
    req.add_header(
        "User-Agent",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    )
    req.add_header(
        "Accept", "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
    )
    req.add_header("Accept-Language", "en-US,en;q=0.5")
    # Don't request compressed encoding to avoid decompression complexity
    req.add_header("Accept-Encoding", "identity")
    req.add_header("Connection", "keep-alive")
    return req


_MAX_RETRIES = 3


def _fetch_url_with_retry(
    url: str, max_retries: int = _MAX_RETRIES
) -> Tuple[bool, Optional[str], str]:
    """
    Fetch a URL with automatic retry logic for transient failures.

    Handles HTTP errors (especially rate limiting with 429 status) and network
    errors by retrying with exponential backoff. Uses browser-like headers
    to avoid blocking.

    Args:
        url: URL to fetch
        max_retries: Maximum number of retry attempts (default: 3)

    Returns:
        Tuple of (success: bool, content: str or None, error_message: str)
    """
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url)
            _add_browser_headers(req)
            with urllib.request.urlopen(req, timeout=8) as response:
                content = response.read().decode("utf-8", errors="ignore")
                return (True, content, "")
        except urllib.error.HTTPError as e:
            if e.code == 429:  # Too Many Requests - retry with backoff
                wait_time = (attempt + 1) * 2
                logger.debug(f"Rate limited, waiting {wait_time}s before retry...")
                time.sleep(wait_time)
                continue
            else:  # Not found, forbidden, gone, etc.
                return (False, None, f"Server returned {e.code}")
        except Exception as e:
            if attempt < max_retries - 1:
                # Retry on network errors with delay
                time.sleep(1)
                continue
    return (False, None, "Fetch failed after all retries")


def _fetch_and_cache_url(
    base_url: str, anchor: Optional[str]
) -> Tuple[bool, Optional[str], str]:
    """
    Fetch URL content and cache it to minimize HTTP requests.

    For GitHub URLs, converts blob URLs to raw.githubusercontent.com URLs
    for more efficient content retrieval. Uses a global cache to avoid
    fetching the same URL multiple times.

    Args:
        base_url: Base URL to fetch (without anchor)
        anchor: Optional anchor/fragment identifier

    Returns:
        Tuple of (link_is_valid: bool, content: str or None, error_message: str)
    """
    # For GitHub file URLs, use raw URL to get content for anchor validation
    raw_url = _get_github_raw_url(base_url, anchor)
    fetch_url = raw_url if raw_url else base_url

    if fetch_url in _url_cache:
        return _url_cache[fetch_url]

    link_is_valid, raw_url_content, error_message = _fetch_url_with_retry(fetch_url)
    _url_cache[fetch_url] = (link_is_valid, raw_url_content, error_message)

    return _url_cache[fetch_url]


def _extract_html_ids_and_names(content: str) -> List[str]:
    """
    Extract all HTML id and name attributes from content.

    Args:
        content: HTML or markdown content to search

    Returns:
        List of lowercase id/name attribute values
    """
    html_ids = re.findall(r'id=["\']([^"\']+)["\']', content, re.IGNORECASE)
    html_names = re.findall(r'name=["\']([^"\']+)["\']', content, re.IGNORECASE)
    return [id_val.lower() for id_val in html_ids + html_names]


def _is_markdown_file(file_path: str) -> bool:
    """
    Check if a file path points to a markdown file.

    Args:
        file_path: Path to check

    Returns:
        True if file exists, is a .md file, and is not excluded
    """
    return (
        os.path.isfile(file_path)
        and os.path.splitext(file_path)[1] == ".md"
        and not is_excluded(file_path)
    )


def _is_github_line_anchor(anchor: Optional[str]) -> bool:
    """
    Check if an anchor is a GitHub line number anchor.

    GitHub supports special anchors like #L123 (single line) or #L10-L20 (line range).
    These anchors don't need validation against content since they're always
    valid if the file exists.

    Args:
        anchor: Anchor string to check (without # prefix)

    Returns:
        True if anchor matches GitHub line number format, False otherwise
    """
    if not anchor:
        return False
    # Matches L123 or L123-L456 (line ranges)
    return bool(re.match(r"^L\d+(-L\d+)?$", anchor))


def _validate_anchor_in_content(
    anchor: Optional[str],
    content: str,
    is_markdown: bool = False,
    slugger: Optional[GithubSlugger] = None,
) -> Tuple[bool, str]:
    """
    Validate that an anchor/fragment identifier exists in the content.

    For markdown files, extracts headings and converts them to GitHub-style
    anchors using the slugger. Also checks HTML id and name attributes.
    For HTML content, searches for id and name attributes directly.

    Args:
        anchor: Anchor string to validate (without # prefix)
        content: File content to search in
        is_markdown: True if content is markdown, False if HTML
        slugger: GithubSlugger instance for converting headings to anchors

    Returns:
        Tuple of (is_valid: bool, error_message: str)
    """
    if not anchor:
        return True, ""

    # GitHub line number anchors are always valid if the file exists
    if _is_github_line_anchor(anchor):
        return True, ""

    # Normalize anchor to lowercase for case-insensitive comparison (GitHub anchors are case-insensitive)
    anchor_lower = anchor.lower()

    if is_markdown:
        # For markdown, check both headings and HTML elements with IDs
        available_anchors = set()

        # 1. Extract headings and convert to GitHub anchor format
        # Handle duplicate headings: GitHub adds -1, -2, etc. for duplicates
        # Match ATX-style headings: ## Heading
        atx_headings = re.findall(r"^#{1,6}\s+(.+?)\s*$", content, re.MULTILINE)
        # Match setext-style headings: Heading\n====== or Heading\n------
        setext_headings = re.findall(r"^(.+?)\s*\n[=-]+\s*$", content, re.MULTILINE)
        headings = atx_headings + setext_headings
        for h in headings:
            expected_anchor = _markdown_heading_to_anchor(h, slugger)
            available_anchors.add(expected_anchor)

        # 2. Extract HTML elements with id and name attributes
        html_ids_and_names = _extract_html_ids_and_names(content)
        available_anchors.update(html_ids_and_names)

        res = anchor_lower in available_anchors
        error_message = "" if res else f"Markdown anchor '#{anchor}' not found"

        return res, error_message
    else:
        # For HTML, check for id or name attributes
        all_ids = _extract_html_ids_and_names(content)

        # Check if anchor matches any id/name attribute (with or without user-content- prefix)
        res = anchor_lower in all_ids or f"user-content-{anchor_lower}" in all_ids
        error_message = "" if res else f"HTML anchor '#{anchor}' not found"
        return res, error_message


def validate_link(link: str, src_doc_path: str) -> Tuple[bool, str]:
    """
    Validate if a URL or relative path exists and is accessible.

    Performs comprehensive validation including:
    - Checking if URLs are reachable (with retry logic)
    - Validating file/directory existence for relative paths
    - Verifying anchor/fragment identifiers in markdown files
    - Handling special cases (mailto:, GitHub line anchors, etc.)

    Uses caching to minimize HTTP requests for the same URLs.

    Args:
        link: URL or relative path to validate
        src_doc_path: Absolute path to the source document containing the link

    Returns:
        Tuple of (is_valid: bool, error_message: str)
        - is_valid: True if link is valid, False otherwise
        - error_message: Empty string if valid, error description if invalid
    """
    if not link or not isinstance(link, str):
        error_message = "Invalid link"
        return False, error_message

    # Skip non-HTTP(S) URLs (mailto:, tel:, etc.)
    if link.startswith("mailto:") or link.startswith("tel:") or link.startswith("ftp:"):
        return True, ""  # Consider these valid (can't validate them)

    # Skip malformed URLs (missing protocol)
    if link.startswith("ttps://") or (
        link.startswith("://") and not link.startswith("http")
    ):
        error_message = "Malformed URL"
        return False, error_message  # Malformed URL

    # Extract anchor from link (common for both URL and relative path)
    # Handle query parameters: split by # first, then by ? to separate query params
    if "#" in link:
        parts = link.split("#", 1)
        base_with_query = parts[0]
        anchor = parts[1] if len(parts) > 1 else None
        # Remove query parameters from base_link for processing
        base_link = base_with_query.split("?")[0]
    else:
        # Remove query parameters even if no anchor
        base_link = link.split("?")[0]
        anchor = None

    # Initialize slugger if we might need it for anchor validation
    slugger: Optional[GithubSlugger] = GithubSlugger() if anchor else None

    # Check if it's a URL
    if type_of_link(link) == LinkType.URL:
        # GitHub line number anchors (e.g., #L123, #L123-L456) are always valid if URL exists
        if anchor and _is_github_line_anchor(anchor):
            # For line anchors, just check if the URL exists (don't need to fetch content)
            # TODO: Add a check to see if the line number is valid
            link_is_valid, _, error_message = _fetch_and_cache_url(base_link, anchor)
            return link_is_valid, error_message

        # Fetch and cache the URL content
        link_is_valid, raw_url_content, error_message = _fetch_and_cache_url(
            base_link, anchor
        )

        if not link_is_valid:
            return False, error_message

        # Validate anchor if present (for GitHub markdown files where we can reliably validate)
        # Other websites use various anchor generation schemes that are hard to predict
        if anchor and raw_url_content and "github.com" in base_link:
            # Check if it's a markdown file or a directory (which would have README.md)
            is_markdown = base_link.endswith(".md") or (
                anchor and not _is_file_url(base_link)
            )
            if is_markdown:
                return _validate_anchor_in_content(
                    anchor, raw_url_content, is_markdown=True, slugger=slugger
                )

        # For other URLs with anchors, consider valid if URL exists
        # (we can't reliably validate anchors on non-GitHub sites)
        return True, ""
    else:
        # Handle same-page anchors (links starting with #)
        if not base_link or base_link.strip() == "":
            # Same-page anchor - validate against current file
            target_path = src_doc_path
        else:
            # It's a relative path - resolve and check if file exists
            target_path = os.path.normpath(
                os.path.join(os.path.dirname(src_doc_path), base_link)
            )

        # Check if file or directory exists
        if not os.path.exists(target_path):
            return False, f"File or directory '{target_path}' not found"

        # Validate anchor if present
        if anchor:
            # GitHub line number anchors (e.g., #L123, #L123-L456) are always valid if file exists
            if _is_github_line_anchor(anchor):
                # For line anchors, file must exist (can be any file type)
                return (
                    os.path.isfile(target_path)
                    or (
                        os.path.isdir(target_path)
                        and os.path.isfile(os.path.join(target_path, "README.md"))
                    ),
                    f"File or directory '{target_path}' not found",
                )

            # Determine which file to read for anchor validation
            file_to_validate = None
            if _is_markdown_file(target_path):
                # Case 1: target is a markdown file
                file_to_validate = target_path
            elif os.path.isdir(target_path):
                # Case 2: target is a directory with README.md
                readme_path = os.path.join(target_path, "README.md")
                if os.path.isfile(readme_path):
                    file_to_validate = readme_path
                else:
                    return False, f"README.md not found in directory '{target_path}'"

            # Validate anchor in the file
            if file_to_validate:
                try:
                    with open(file_to_validate, "r", encoding="utf-8") as f:
                        raw_file_content = f.read()
                except Exception as e:
                    return (
                        False,
                        f"Error reading file '{file_to_validate}' for anchor validation: {e}",
                    )
                return _validate_anchor_in_content(
                    anchor, raw_file_content, is_markdown=True, slugger=slugger
                )

        return True, ""


def _is_in_code_block(match_start: int, content: str) -> bool:
    """
    Determine if a character position is inside a fenced code block.

    Checks if the given position in the content is within a markdown
    fenced code block (using ``` or ~~~ delimiters). This is used to
    skip processing hyperlinks that appear in code examples.

    Args:
        match_start: Character position in content to check
        content: Full document content

    Returns:
        True if position is inside a code block, False otherwise
    """
    # Find all code block ranges
    code_block_ranges = []

    # Find fenced code blocks (``` or ~~~)
    lines = content.split("\n")
    in_fenced_block = False
    fence_start = None
    fence_char = None  # Track which fence character we're using

    for i, line in enumerate(lines):
        stripped = line.strip()
        # Check for fenced code block start/end
        if stripped.startswith("```"):
            if not in_fenced_block:
                # Opening fence
                in_fenced_block = True
                fence_start = i
                fence_char = "`"
            elif fence_char == "`":
                # Closing fence (same type)
                code_block_ranges.append((fence_start, i))
                in_fenced_block = False
                fence_start = None
                fence_char = None
        elif stripped.startswith("~~~"):
            if not in_fenced_block:
                # Opening fence
                in_fenced_block = True
                fence_start = i
                fence_char = "~"
            elif fence_char == "~":
                # Closing fence (same type)
                code_block_ranges.append((fence_start, i))
                in_fenced_block = False
                fence_start = None
                fence_char = None

    # Close any open fenced blocks at end of file
    if in_fenced_block and fence_start is not None:
        code_block_ranges.append((fence_start, len(lines) - 1))

    # Check if match_start is inside any code block
    match_line = content[:match_start].count("\n")
    for start_line, end_line in code_block_ranges:
        if start_line <= match_line <= end_line:
            return True

    return False


def replace_hyperlink(m: re.Match, src_doc_path: str, content: str) -> str:
    """
    Process and replace a single hyperlink match in markdown content.

    This function is called by regex substitution to process each hyperlink found.
    It converts URLs to relative paths (when appropriate) and relative paths to
    URLs (when needed), then validates the resulting link. Links inside code
    blocks are skipped to avoid modifying code examples.

    Args:
        m: Regex match object containing the hyperlink
        src_doc_path: Absolute path to the source document file
        content: Full content of the source document (to check for code blocks)

    Returns:
        Modified hyperlink string (or original if in code block)

    Note:
        Validation errors are collected in global _validation_errors list
        TODO: Support HTML tags for future docs (e.g., <a href=...>).
    """
    global _validation_errors

    # Skip links inside code blocks
    match_start = m.start()
    if _is_in_code_block(match_start, content):
        return m.group(0)  # Return original without modification

    hyperlink_str = m.group(2)

    # Convert to the appropriate format, e.g. relative path if internal or URL if external.
    if type_of_link(hyperlink_str) == LinkType.URL:
        res = replace_url_with_relpath(hyperlink_str, src_doc_path)
    else:
        res = replace_relpath_with_url(hyperlink_str, src_doc_path)

    # Validate the resulting URL or relative path exists
    link_is_valid, error_message = validate_link(res, src_doc_path)
    if not link_is_valid:
        if not any(domain in res for domain in white_list_domains):
            _validation_errors.append((res, hyperlink_str, src_doc_path, error_message))

    return m.group(1) + res + m.group(3)


def report_validation_errors() -> None:
    """
    Display a formatted report of all link validation errors.

    Prints a colorized summary table showing all invalid links found during
    preprocessing, including the invalid URL, original link, source file,
    and error message. If no errors were found, displays a success message.

    Uses ANSI color codes for better readability in terminal output.
    """
    global _validation_errors
    if not _validation_errors:
        logger.info("✅ All hyperlinks validated successfully.")
        return

    # ANSI color codes
    RED = "\033[91m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    BOLD = "\033[1m"
    RESET = "\033[0m"

    print()
    print(f"{RED}{BOLD}{'═' * 80}{RESET}")
    print(f"{RED}{BOLD}  ❌ HYPERLINK VALIDATION REPORT{RESET}")
    print(f"{RED}{BOLD}{'═' * 80}{RESET}")
    print(
        f"{WHITE}  Found {YELLOW}{len(_validation_errors)}{WHITE} invalid hyperlink(s){RESET}"
    )
    print(f"{RED}{'─' * 80}{RESET}")

    for idx, (url, original, src_path, error_message) in enumerate(
        _validation_errors, 1
    ):
        print()
        print(f"{YELLOW}{BOLD}  [{idx}]{RESET}")
        print(f"{WHITE}  ├─ URL:      {CYAN}{url}{RESET}")
        print(f"{WHITE}  ├─ Original: {CYAN}{original}{RESET}")
        print(f"{WHITE}  └─ File:     {CYAN}{src_path}{RESET}")
        print(f"{WHITE}  └─ Error:    {RED}{error_message}{RESET}")
    print()
    print(f"{RED}{BOLD}{'═' * 80}{RESET}")
    print()


def preprocess_docs(
    exclude_paths: Optional[List[str]] = None,
) -> Dict[str, Union[float, int]]:
    """
    Find all markdown files and preprocess their hyperlinks.

    This is the main preprocessing function that:
    1. Discovers all .md files in the docs directory
    2. Processes each file to convert and validate hyperlinks
    3. Reports validation errors at the end

    Hyperlinks are converted from GitHub URLs to relative paths (when appropriate)
    or from relative paths to GitHub URLs (when needed). All links are validated
    to ensure they exist and are accessible.

    Args:
        exclude_paths: Optional list of paths to exclude (currently unused)

    Returns:
        Dictionary containing timing information for preprocessing steps:
        - find_files: Time to discover markdown files
        - process_docs: Time to process all documents
        - processed_count: Number of documents processed
        - report_errors: Time to report validation errors
        - total_files: Total number of markdown files found
    """
    global _validation_errors
    # Clear previous errors
    _validation_errors.clear()

    # Find all ".md" files
    cmd = f"find {server_docs_abspath} -name '*.md'"
    result = subprocess.run(cmd, check=True, capture_output=True, text=True, shell=True)
    docs_list = [path for path in result.stdout.split("\n") if path]

    # Read, preprocess and write back to each document file
    for doc_abspath in docs_list:
        if is_excluded(doc_abspath):
            continue

        logger.info(f"Preprocessing {doc_abspath}")
        with open(doc_abspath) as f:
            content = f.read()

        content = hyperlink_reg.sub(
            lambda m: replace_hyperlink(m, doc_abspath, content),
            content,
        )

        with open(doc_abspath, "w") as f:
            f.write(content)


def main() -> None:
    """
    Main entry point for the documentation generation script.

    Orchestrates the complete documentation setup process:
    1. Reads the repository list from repositories.txt
    2. Clones all required repositories from GitHub
    3. Preprocesses all markdown files to convert and validate hyperlinks
    4. Displays a comprehensive timing summary at the end

    The script tracks timing for each major step and per-repository cloning
    to help identify performance bottlenecks.

    Environment Variables:
        SERVER_ABSPATH: Absolute path to server repository (default: current directory)
        TRITON_SERVER_REPO_TAG: Default repository tag/branch (default: "main")
        TRITON_SERVER_DOCS_LOG_FILE: Log file path (default: "/tmp/docs.log")
        TRITON_SERVER_REPO_ORG: GitHub organization URL

    Command Line Arguments:
        --repo-tag: Repository tag/branch to clone (default: from env or "main")
        --log-file: Path to log file (default: from env or "/tmp/docs.log")
        --repo-file: File listing repositories to clone (default: "repositories.txt")
        --github-organization: GitHub organization URL
    """
    script_start_time = time.time()
    logger.info("Starting setup Triton Server documentation for Sphinx build...")

    # Step 1: Read repository list
    step1_start_time = time.time()
    logger.info(f"Collecting repositories from {args.repo_file}...")
    os.chdir(server_docs_abspath)

    with open(args.repo_file) as f:
        repository_list = f.read().strip().split("\n")
    step1_elapsed = time.time() - step1_start_time

    # Step 2: Clone repositories - track timing for each repo
    step2_start_time = time.time()
    logger.info(f"Starting to clone {len(repository_list)} repositories...")

    for repository in repository_list:
        # clean up previous cloned repositories
        run_command(f"rm -rf {repository}")
        clone_from_github(repository, args.repo_tag, args.github_organization)

    step2_elapsed = time.time() - step2_start_time

    # Step 3: Preprocess documents after all repos are cloned
    step3_start_time = time.time()
    logger.info("Starting document preprocessing...")
    preprocess_docs()
    step3_elapsed = time.time() - step3_start_time

    # Calculate total time
    total_elapsed = time.time() - script_start_time

    # Display comprehensive timing summary
    logger.info("")
    logger.info("=" * 80)
    logger.info("⏱️  TIME USAGE SUMMARY")
    logger.info("=" * 80)
    logger.info("")

    # Overall summary
    logger.info("=" * 80)
    logger.info("OVERALL SUMMARY")
    logger.info("=" * 80)
    logger.info(
        f"  Step 1 - Reading repository list:     {step1_elapsed:>8.2f} seconds ({step1_elapsed/total_elapsed*100:.1f}%)"
    )
    logger.info(
        f"  Step 2 - Cloning repositories:         {step2_elapsed:>8.2f} seconds ({step2_elapsed/total_elapsed*100:.1f}%)"
    )
    logger.info(
        f"  Step 3 - Preprocessing documents:      {step3_elapsed:>8.2f} seconds ({step3_elapsed/total_elapsed*100:.1f}%)"
    )
    logger.info(f"  {'-' * 76}")
    logger.info(
        f"  Total execution time:                  {total_elapsed:>8.2f} seconds ({total_elapsed/60:.2f} minutes)"
    )
    logger.info("=" * 80)
    logger.info("")

    # Report all validation errors at the end
    report_validation_errors()

    # Exit with error code if validation errors were found
    global _validation_errors
    if len(_validation_errors) > 0:
        logger.error(
            f"Exiting with error code 1 due to {len(_validation_errors)} invalid hyperlink(s)"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()

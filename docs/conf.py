#!/usr/bin/env python3

# Copyright 2023-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

import json
import os
import re
from datetime import date

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import httplib2
import nvidia_sphinx_theme
from docutils import nodes
from packaging.version import Version
from sphinx import search

# import sys
# sys.path.insert(0, os.path.abspath('.'))

# -- conf.py setup -----------------------------------------------------------

# conf.py needs to be run in the top level 'docs'
# directory but the calling build script needs to
# be called from the current working directory. We
# change to the 'docs' dir here and then revert back
# at the end of the file.
# current_dir = os.getcwd()
# os.chdir("docs")

# -- Project information -----------------------------------------------------

project = "NVIDIA Triton Inference Server"
copyright = "2018-{}, NVIDIA Corporation".format(date.today().year)
author = "NVIDIA"

# Get the version of Triton this is building.
version_long = "0.0.0"
with open("../TRITON_VERSION") as f:
    version_long = f.readline()
    version_long = version_long.strip()

version_short = re.match(r"^[\d]+\.[\d]+\.[\d]+", version_long).group(0)
version_short_split = version_short.split(".")
one_before = f"{version_short_split[0]}.{int(version_short_split[1]) - 1}.{version_short_split[2]}"


# maintain left-side bar toctrees in `contents` file
# so it doesn't show up needlessly in the index page
master_doc = "contents"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "ablog",
    "myst_parser",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx-prompt",
    # "sphinxcontrib.bibtex",
    "sphinx_tabs.tabs",
    "sphinx_sitemap",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.ifconfig",
    "sphinx.ext.extlinks",
]

suppress_warnings = ["myst.domains", "ref.ref", "myst.header"]

source_suffix = [".rst", ".md"]

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "private-members": True,
}

autosummary_generate = True
autosummary_mock_imports = [
    "tritonclient.grpc.model_config_pb2",
    "tritonclient.grpc.service_pb2",
    "tritonclient.grpc.service_pb2_grpc",
]

napoleon_include_special_with_doc = True

numfig = True

# final location of docs for seo/sitemap
html_baseurl = (
    "https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/"
)

myst_enable_extensions = [
    "dollarmath",
    "amsmath",
    "deflist",
    # "html_admonition",
    "html_image",
    "colon_fence",
    # "smartquotes",
    "replacements",
    # "linkify",
    "substitution",
]
myst_heading_anchors = 5

# Add any paths that contain templates here, relative to this directory.
# templates_path = ["_templates"] # disable it for nvidia-sphinx-theme to show footer

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclusions = None
with open("exclusions.txt", "r") as f:
    exclusions = f.read()
    f.close()
exclude_patterns = exclusions.strip().split("\n")
print(f"exclude_patterns: {exclude_patterns}")

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "nvidia_sphinx_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
# html_css_files = ["custom.css"] # Not needed with new theme

html_theme_options = {
    "collapse_navigation": False,
    "github_url": "https://github.com/triton-inference-server/server",
    "switcher": {
        # use for local testing
        # "json_url": "http://localhost:8000/_static/switcher.json",
        "json_url": "https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/_static/switcher.json",
        "version_match": one_before if "dev" in version_long else version_short,
    },
    "navbar_start": ["navbar-logo", "version-switcher"],
    "primary_sidebar_end": [],
}

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
html_theme_options.update(
    {
        "collapse_navigation": False,
    }
)

deploy_ngc_org = "nvidia"
deploy_ngc_team = "triton"
myst_substitutions = {
    "VersionNum": version_short,
    "deploy_ngc_org_team": f"{deploy_ngc_org}/{deploy_ngc_team}"
    if deploy_ngc_team
    else deploy_ngc_org,
}


def ultimateReplace(app, docname, source):
    result = source[0]
    for key in app.config.ultimate_replacements:
        result = result.replace(key, app.config.ultimate_replacements[key])
    source[0] = result


# this is a necessary hack to allow us to fill in variables that exist in code blocks
ultimate_replacements = {
    "{VersionNum}": version_short,
    "{SamplesVersionNum}": version_short,
    "{NgcOrgTeam}": f"{deploy_ngc_org}/{deploy_ngc_team}"
    if deploy_ngc_team
    else deploy_ngc_org,
}

# bibtex_bibfiles = ["references.bib"]
# To test that style looks good with common bibtex config
# bibtex_reference_style = "author_year"
# bibtex_default_style = "plain"

### We currently use Myst: https://myst-nb.readthedocs.io/en/latest/use/execute.html
nb_execution_mode = "off"  # Global execution disable
# execution_excludepatterns = ['tutorials/tts-python-basics.ipynb']  # Individual notebook disable

###############################
# SETUP SWITCHER
###############################
switcher_path = os.path.join(html_static_path[0], "switcher.json")
versions = []
# Triton 2 releases
correction = -1 if "dev" in version_long else 0
upper_bound = version_short.split(".")[1]
for i in range(2, int(version_short.split(".")[1]) + correction):
    versions.append((f"2.{i}.0", f"triton-inference-server-2{i}0"))

# Triton 1 releases
for i in range(0, 15):
    versions.append((f"1.{i}.0", f"tensorrt_inference_server_1{i}0"))

# Triton Beta Releases
for i in range(1, 11):
    versions.append((f"0.{i}.0_beta", f"inference_server_0{i}0_beta"))

# Patch releases
# Add here.

versions = sorted(versions, key=lambda v: Version(v[0]), reverse=True)

# Build switcher data
json_data = []
for v in versions:
    json_data.append(
        {
            "name": v[0],
            "version": v[0],
            "url": f"https://docs.nvidia.com/deeplearning/triton-inference-server/archives/{v[1]}/user-guide/docs",
        }
    )
if "dev" in version_long:
    json_data.insert(
        0,
        {
            "name": f"{one_before} (current_release)",
            "version": f"{one_before}",
            "url": "https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/index.html",
        },
    )
else:
    json_data.insert(
        0,
        {
            "name": f"{version_short} (current release)",
            "version": f"{version_short}",
            "url": "https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/index.html",
        },
    )

# Trim to last N releases.
json_data = json_data[0:12]

json_data.append(
    {
        "name": "older releases",
        "version": "archives",
        "url": "https://docs.nvidia.com/deeplearning/triton-inference-server/archives/",
    }
)

# validate the links
for i, d in enumerate(json_data):
    h = httplib2.Http()
    resp = h.request(d["url"], "HEAD")
    if int(resp[0]["status"]) >= 400:
        print(d["url"], "NOK", resp[0]["status"])
        exit(1)

# Write switcher data to file
with open(switcher_path, "w") as f:
    json.dump(json_data, f, ensure_ascii=False, indent=4)


def setup(app):
    app.add_config_value("ultimate_replacements", {}, True)
    app.connect("source-read", ultimateReplace)
    app.add_js_file("https://js.hcaptcha.com/1/api.js")

    visitor_script = (
        "//assets.adobedtm.com/5d4962a43b79/c1061d2c5e7b/launch-191c2462b890.min.js"
    )

    if visitor_script:
        app.add_js_file(visitor_script)

    # if not os.environ.get("READTHEDOCS") and not os.environ.get("GITHUB_ACTIONS"):
    #     app.add_css_file(
    #         "https://assets.readthedocs.org/static/css/readthedocs-doc-embed.css"
    #     )
    #     app.add_css_file("https://assets.readthedocs.org/static/css/badge_only.css")

    #     # Create the dummy data file so we can link it
    #     # ref: https://github.com/readthedocs/readthedocs.org/blob/bc3e147770e5740314a8e8c33fec5d111c850498/readthedocs/core/static-src/core/js/doc-embed/footer.js  # noqa: E501
    #     app.add_js_file("rtd-data.js")
    #     app.add_js_file(
    #         "https://assets.readthedocs.org/static/javascript/readthedocs-doc-embed.js",
    #         priority=501,
    #     )


# cleanup
# os.chdir(current_dir)

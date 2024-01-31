#!/usr/bin/env python3

# Copyright 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

from docutils import nodes
from sphinx import addnodes, package_dir, search
from sphinx.util import split_into

sys.path.insert(
    0, os.path.abspath("/usr/local/lib/python3.10/dist-packages/tritonserver")
)

# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = "NVIDIA Triton Inference Server"
copyright = "2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved"
author = "NVIDIA"

# The full version, including alpha/beta/rc tags
# Env only set during riva-release process, otherwise keep as dev for all internal builds
release = os.getenv("TRITON_VERSION", "dev")

# maintain left-side bar toctrees in `contents` file
# so it doesn't show up needlessly in the index page
master_doc = "contents"

# -- Autodoc configuration ---------------------------------------------------
autodoc_class_signature = "separated"
autodoc_default_options = {"members": True}
default_role = "any"
autodoc_typehints_format = "short"
python_use_unqualified_type_names = True

# Rename module from internal modules
# Due to how we are aliasing objects from the bindings
# We have to fix up the naming for documentation
class_names = [
    "LogFormat",
    "ModelLoadDeviceLimit",
    "MemoryType",
    "DataType",
    "LogLevel",
    "MetricFormat",
    "ModelControlMode",
    "RateLimitMode",
    "RateLimiterResource",
]

type_aliases = {name: name for name in class_names}
napoleon_type_aliases = type_aliases
autodoc_type_aliases = type_aliases

import tritonserver

methods_to_skip = []
for class_name in class_names:
    _class = getattr(tritonserver, class_name)
    _class.__module__ = "tritonserver"
    _class.__name__ = class_name
    init_method = getattr(_class, "__init__", None)
    name_method = getattr(_class, "name", None)
    if init_method:
        methods_to_skip.append(init_method)
    if name_method:
        methods_to_skip.append(name_method)


def autodoc_skip_member(app, what, name, obj, skip, options):
    if obj in methods_to_skip:
        return True


def autodoc_before_process_signature(app, obj, bound_method):
    for key, annotation in obj.__annotations__.items():
        name = getattr(annotation, "__name__", None)
        if name and name == "LogLevel":
            obj.__annotations__[key] = "tritonserver.LogLevel"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "ablog",
    "myst_nb",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx-prompt",
    # "sphinxcontrib.bibtex",
    "sphinx_tabs.tabs",
    "sphinx_sitemap",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_markdown_builder",
]

suppress_warnings = ["myst.domains", "ref.ref"]

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
    # "html_image",
    "colon_fence",
    # "smartquotes",
    "replacements",
    # "linkify",
    "substitution",
]
myst_heading_anchors = 5

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["README.md"]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_book_theme"
html_logo = "_static/nvidia-logo-horiz-rgb-blk-for-screen.png"
html_title = "NVIDIA Triton Inference Server"
html_short_title = "Triton"
html_copy_source = True
html_sourcelink_suffix = ""
html_favicon = "_static/nvidia-logo-vert-rgb-blk-for-screen.png"
html_last_updated_fmt = ""
html_additional_files = ["index.html"]

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_css_files = ["custom.css"]

html_theme_options = {
    "path_to_docs": "docs",
    # "launch_buttons": {
    #     "binderhub_url": "https://mybinder.org",
    #     "colab_url": "https://colab.research.google.com/",
    #     "deepnote_url": "https://deepnote.com/",
    #     "notebook_interface": "jupyterlab",
    #     "thebe": True,
    #     # "jupyterhub_url": "https://datahub.berkeley.edu",  # For testing
    # },
    "use_edit_page_button": False,
    "use_issues_button": True,
    "use_repository_button": True,
    "use_download_button": False,
    "logo_only": False,
    "show_toc_level": 2,
    "extra_navbar": "",
    "extra_footer": "",
    "repository_url": "https://github.com/triton-inference-server/server",
    "use_repository_button": True,
}

version_short = release
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
jupyter_execute_notebooks = "off"  # Global execution disable
# execution_excludepatterns = ['tutorials/tts-python-basics.ipynb']  # Individual notebook disable


def autodoc_process_docstring(app, what, name, obj, options, lines):
    print(what)
    print(name)
    print(obj)


def setup(app):
    app.add_config_value("ultimate_replacements", {}, True)
    app.connect("source-read", ultimateReplace)
    app.connect("autodoc-before-process-signature", autodoc_before_process_signature)
    app.connect("autodoc-skip-member", autodoc_skip_member)
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


# Patch for sphinx.search stemming short terms (i.e. tts -> tt)
# https://github.com/sphinx-doc/sphinx/blob/4.5.x/sphinx/search/__init__.py#L380
def sphinxSearchIndexFeed(
    self, docname: str, filename: str, title: str, doctree: nodes.document
):
    """Feed a doctree to the index."""
    self._titles[docname] = title
    self._filenames[docname] = filename

    visitor = search.WordCollector(doctree, self.lang)
    doctree.walk(visitor)

    # memoize self.lang.stem
    def stem(word: str) -> str:
        try:
            return self._stem_cache[word]
        except KeyError:
            self._stem_cache[word] = self.lang.stem(word).lower()
            return self._stem_cache[word]

    _filter = self.lang.word_filter

    self._all_titles[docname] = visitor.found_titles

    for word in visitor.found_title_words:
        stemmed_word = stem(word)
        if len(stemmed_word) > 3 and _filter(stemmed_word):
            self._title_mapping.setdefault(stemmed_word, set()).add(docname)
        elif _filter(word):  # stemmer must not remove words from search index
            self._title_mapping.setdefault(word.lower(), set()).add(docname)

    for word in visitor.found_words:
        stemmed_word = stem(word)
        # again, stemmer must not remove words from search index
        if len(stemmed_word) <= 3 or not _filter(stemmed_word) and _filter(word):
            stemmed_word = word.lower()
        already_indexed = docname in self._title_mapping.get(stemmed_word, set())
        if _filter(stemmed_word) and not already_indexed:
            self._mapping.setdefault(stemmed_word, set()).add(docname)

    # find explicit entries within index directives
    _index_entries: Set[Tuple[str, str, str]] = set()
    for node in doctree.findall(addnodes.index):
        for entry_type, value, tid, main, *index_key in node["entries"]:
            tid = tid or ""
            try:
                if entry_type == "single":
                    try:
                        entry, subentry = split_into(2, "single", value)
                    except ValueError:
                        (entry,) = split_into(1, "single", value)
                        subentry = ""
                    _index_entries.add((entry, tid, main))
                    if subentry:
                        _index_entries.add((subentry, tid, main))
                elif entry_type == "pair":
                    first, second = split_into(2, "pair", value)
                    _index_entries.add((first, tid, main))
                    _index_entries.add((second, tid, main))
                elif entry_type == "triple":
                    first, second, third = split_into(3, "triple", value)
                    _index_entries.add((first, tid, main))
                    _index_entries.add((second, tid, main))
                    _index_entries.add((third, tid, main))
                elif entry_type in {"see", "seealso"}:
                    first, second = split_into(2, "see", value)
                    _index_entries.add((first, tid, main))
            except ValueError:
                pass

    self._index_entries[docname] = sorted(_index_entries)


search.IndexBuilder.feed = sphinxSearchIndexFeed

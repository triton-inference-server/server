from tritonfrontend._c.tritonfrontend_bindings import setLoggingVerbose


def enable_logging(level):
    setLoggingVerbose(level)

def test_package_can_be_imported() -> None:
    import marketplace_agent

    assert marketplace_agent.__name__ == "marketplace_agent"
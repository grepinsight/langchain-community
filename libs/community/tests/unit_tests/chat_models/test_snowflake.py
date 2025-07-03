"""Test ChatSnowflakeCortex."""

import os
from unittest.mock import Mock, patch

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from langchain_community.chat_models.snowflake import (
    ChatSnowflakeCortex,
    _convert_message_to_dict,
)


def test_messages_to_prompt_dict_with_valid_messages() -> None:
    messages = [
        SystemMessage(content="System Prompt"),
        HumanMessage(content="User message #1"),
        AIMessage(content="AI message #1"),
        HumanMessage(content="User message #2"),
        AIMessage(content="AI message #2"),
    ]
    result = [_convert_message_to_dict(m) for m in messages]
    expected = [
        {"role": "system", "content": "System Prompt"},
        {"role": "user", "content": "User message #1"},
        {"role": "assistant", "content": "AI message #1"},
        {"role": "user", "content": "User message #2"},
        {"role": "assistant", "content": "AI message #2"},
    ]
    assert result == expected


@patch("langchain_community.chat_models.snowflake.Session")
def test_chat_snowflake_cortex_authenticator_parameter(
    mock_session_class: Mock,
) -> None:
    """Test ChatSnowflakeCortex handles authenticator parameter."""
    # Mock the Session creation
    mock_session = Mock()
    mock_session_class.builder.configs.return_value.create.return_value = mock_session

    # Set required environment variables
    env_vars = {
        "SNOWFLAKE_ACCOUNT": "test_account",
        "SNOWFLAKE_USERNAME": "test_user",
        "SNOWFLAKE_PASSWORD": "test_password",
        "SNOWFLAKE_DATABASE": "test_db",
        "SNOWFLAKE_SCHEMA": "test_schema",
        "SNOWFLAKE_WAREHOUSE": "test_warehouse",
        "SNOWFLAKE_ROLE": "test_role",
    }

    with patch.dict(os.environ, env_vars, clear=False):
        # Test with authenticator parameter
        chat = ChatSnowflakeCortex(authenticator="username_password_mfa")

        # Verify the session was created with authenticator
        mock_session_class.builder.configs.assert_called_once()
        connection_params = mock_session_class.builder.configs.call_args[0][0]

        assert "authenticator" in connection_params
        assert connection_params["authenticator"] == "username_password_mfa"
        assert chat.snowflake_authenticator == "username_password_mfa"


@patch("langchain_community.chat_models.snowflake.Session")
def test_chat_snowflake_cortex_authenticator_env_var(mock_session_class: Mock) -> None:
    """Test ChatSnowflakeCortex handles authenticator from environment variable."""
    # Mock the Session creation
    mock_session = Mock()
    mock_session_class.builder.configs.return_value.create.return_value = mock_session

    # Set required environment variables including authenticator
    env_vars = {
        "SNOWFLAKE_ACCOUNT": "test_account",
        "SNOWFLAKE_USERNAME": "test_user",
        "SNOWFLAKE_PASSWORD": "test_password",
        "SNOWFLAKE_DATABASE": "test_db",
        "SNOWFLAKE_SCHEMA": "test_schema",
        "SNOWFLAKE_WAREHOUSE": "test_warehouse",
        "SNOWFLAKE_ROLE": "test_role",
        "SNOWFLAKE_AUTHENTICATOR": "oauth",
    }

    with patch.dict(os.environ, env_vars, clear=False):
        # Test without explicit authenticator parameter
        chat = ChatSnowflakeCortex()

        # Verify the session was created with authenticator from env var
        mock_session_class.builder.configs.assert_called_once()
        connection_params = mock_session_class.builder.configs.call_args[0][0]

        assert "authenticator" in connection_params
        assert connection_params["authenticator"] == "oauth"
        assert chat.snowflake_authenticator == "oauth"


@patch("langchain_community.chat_models.snowflake.Session")
def test_chat_snowflake_cortex_no_authenticator(mock_session_class: Mock) -> None:
    """Test ChatSnowflakeCortex works without authenticator parameter."""
    # Mock the Session creation
    mock_session = Mock()
    mock_session_class.builder.configs.return_value.create.return_value = mock_session

    # Set required environment variables without authenticator
    env_vars = {
        "SNOWFLAKE_ACCOUNT": "test_account",
        "SNOWFLAKE_USERNAME": "test_user",
        "SNOWFLAKE_PASSWORD": "test_password",
        "SNOWFLAKE_DATABASE": "test_db",
        "SNOWFLAKE_SCHEMA": "test_schema",
        "SNOWFLAKE_WAREHOUSE": "test_warehouse",
        "SNOWFLAKE_ROLE": "test_role",
    }

    with patch.dict(os.environ, env_vars, clear=False):
        # Test without authenticator parameter
        chat = ChatSnowflakeCortex()

        # Verify the session was created without authenticator
        mock_session_class.builder.configs.assert_called_once()
        connection_params = mock_session_class.builder.configs.call_args[0][0]

        assert "authenticator" not in connection_params
        assert chat.snowflake_authenticator is None


@patch("langchain_community.chat_models.snowflake.Session")
def test_chat_snowflake_cortex_authenticator_alias(mock_session_class: Mock) -> None:
    """Test ChatSnowflakeCortex handles authenticator alias."""
    # Mock the Session creation
    mock_session = Mock()
    mock_session_class.builder.configs.return_value.create.return_value = mock_session

    # Set required environment variables
    env_vars = {
        "SNOWFLAKE_ACCOUNT": "test_account",
        "SNOWFLAKE_USERNAME": "test_user",
        "SNOWFLAKE_PASSWORD": "test_password",
        "SNOWFLAKE_DATABASE": "test_db",
        "SNOWFLAKE_SCHEMA": "test_schema",
        "SNOWFLAKE_WAREHOUSE": "test_warehouse",
        "SNOWFLAKE_ROLE": "test_role",
    }

    with patch.dict(os.environ, env_vars, clear=False):
        # Test with authenticator alias
        chat = ChatSnowflakeCortex(authenticator="snowflake")

        # Verify the session was created with authenticator
        mock_session_class.builder.configs.assert_called_once()
        connection_params = mock_session_class.builder.configs.call_args[0][0]

        assert "authenticator" in connection_params
        assert connection_params["authenticator"] == "snowflake"
        assert chat.snowflake_authenticator == "snowflake"

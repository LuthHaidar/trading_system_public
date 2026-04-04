import pytest
from pydantic import ValidationError

from utils.config_schema import _validate_model, validate_main_config, validate_strategies_config


def _valid_config():
    return {
        'portfolio': {'initial_capital': 10000, 'currency': 'USD'},
        'execution': {'rebalance_timeframe': 'weekly', 'timing': 'close'},
        'risk': {
            'position_sizing_method': 'equal',
            'max_position_size': 0.2,
            'min_position_size': 0.01,
            'volatility_window': 60,
        },
        'data': {'data_dir': './data', 'max_stale_price_days': 5},
        'ibkr': {'host': '127.0.0.1', 'port': 7497, 'client_id': 1},
    }


def test_valid_minimal_config_passes():
    cfg = _valid_config()
    parsed = validate_main_config(cfg)
    assert parsed.portfolio.initial_capital == 10000


def test_unknown_top_level_key_fails():
    cfg = _valid_config()
    cfg['unknown_section'] = {'foo': 'bar'}
    with pytest.raises(ValidationError):
        validate_main_config(cfg)


def test_wrong_type_fails():
    cfg = _valid_config()
    cfg['portfolio']['initial_capital'] = 'foo'
    with pytest.raises(ValidationError):
        validate_main_config(cfg)


def test_missing_required_field_fails():
    cfg = _valid_config()
    del cfg['ibkr']['host']
    with pytest.raises(ValidationError):
        validate_main_config(cfg)


def test_dead_execution_frequency_key_fails():
    cfg = _valid_config()
    cfg['execution']['frequency'] = 'daily'
    with pytest.raises(ValidationError):
        validate_main_config(cfg)


def test_dead_data_update_hour_key_fails():
    cfg = _valid_config()
    cfg['data']['update_hour'] = 17
    with pytest.raises(ValidationError):
        validate_main_config(cfg)


def test_invalid_strategy_stanza_type_fails():
    with pytest.raises(TypeError):
        validate_strategies_config({'momentum': ['not', 'a', 'mapping']})


def test_valid_strategy_stanza_passes():
    parsed = validate_strategies_config({'momentum': {'enabled': True, 'lookback': 126}})
    assert parsed['momentum'].enabled is True




def test_validate_model_supports_parse_obj_only_classes():
    class ParseObjOnly:
        called = False

        @classmethod
        def parse_obj(cls, payload):
            cls.called = True
            return payload

    payload = {'ok': True}
    parsed = _validate_model(ParseObjOnly, payload)
    assert ParseObjOnly.called is True
    assert parsed == payload

def test_contract_overrides_config_passes():
    cfg = _valid_config()
    cfg['ibkr']['contract_overrides'] = {
        'SPY': {'symbol': 'SPY', 'exchange': 'SMART', 'currency': 'USD', 'primary_exchange': 'ARCA'}
    }
    parsed = validate_main_config(cfg)
    assert parsed.ibkr.contract_overrides['SPY'].primary_exchange == 'ARCA'


def test_contract_overrides_extra_field_fails():
    cfg = _valid_config()
    cfg['ibkr']['contract_overrides'] = {
        'SPY': {'symbol': 'SPY', 'exchange': 'SMART', 'currency': 'USD', 'foo': 'bar'}
    }
    with pytest.raises(ValidationError):
        validate_main_config(cfg)


def test_cross_field_min_forward_must_cover_max_stale():
    cfg = _valid_config()
    cfg['execution']['min_forward_data_days'] = 2
    cfg['data']['max_stale_price_days'] = 5
    with pytest.raises(ValueError):
        validate_main_config(cfg)

def test_kelly_fraction_bounds_enforced():
    cfg = _valid_config()
    cfg['risk']['kelly_fraction'] = 0.0
    with pytest.raises(ValidationError):
        validate_main_config(cfg)

    cfg = _valid_config()
    cfg['risk']['kelly_fraction'] = 1.1
    with pytest.raises(ValidationError):
        validate_main_config(cfg)

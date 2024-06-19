# model parameters for the v1 version of machine learning. Includes original and detrended data
parameters = {
    'AUS': {
        'detrended': {
            'drift': {
                'lgb': {
                    'num_leaves': 25,
                    'learning_rate': 0.01,
                    'max_depth': 10,
                    'n_estimators': 610,
                    'min_child_samples': 39,
                    'reg_alpha': 0.4,
                    'reg_lambda': 0.1,
                },
                'xgb': {
                    'eta': 0.09,
                    'max_depth': 6,
                    'n_estimators': 950,
                    'min_child_weight': 11,
                    'reg_alpha': 0.5,
                    'reg_lambda': 0.6,
                },
            },
            'diffusion': {
                'lgb': {
                    'num_leaves': 22,
                    'learning_rate': 0.14,
                    'max_depth': 11,
                    'n_estimators': 550,
                    'min_child_samples': 48,
                    'reg_alpha': 0.0004,
                    'reg_lambda': 0.9,
                },
                'xgb': {
                    'eta': 0.42,
                    'max_depth': 6,
                    'n_estimators': 510,
                    'min_child_weight': 24,
                    'reg_alpha': 0.0,
                    'reg_lambda': 0.3,
                },
            },
        },
        'original': {
            'drift': {
                'lgb': {
                    'num_leaves': 38,
                    'learning_rate': 0.012,
                    'max_depth': 4,
                    'n_estimators': 405,
                    'min_child_samples': 44,
                    'reg_alpha': 0.9,
                    'reg_lambda': 0.5,
                },
                'xgb': {
                    'eta': 0.01,
                    'max_depth': 3,
                    'n_estimators': 324,
                    'min_child_weight': 2,
                    'reg_alpha': 0.1,
                    'reg_lambda': 0.4,
                },
            },
            'diffusion': {
               'lgb': {
                    'num_leaves': 29,
                    'learning_rate': 0.17,
                    'max_depth': 11,
                    'n_estimators': 168,
                    'min_child_samples': 24,
                    'reg_alpha': 0.001,
                    'reg_lambda': 0.4,
                },
               'xgb': {
                    'eta': 0.42,
                    'max_depth': 6,
                    'n_estimators': 510,
                    'min_child_weight': 24,
                    'reg_alpha': 0.0005,
                    'reg_lambda': 0.3,
                },
            },
        }
    },
    'CE': {
        'detrended': {
            'drift': {
                'lgb': {
                    'num_leaves': 14,
                    'learning_rate': 0.03,
                    'max_depth': 5,
                    'n_estimators': 873,
                    'min_child_samples': 75,
                    'reg_alpha': 0.1,
                    'reg_lambda': 1.0,
                },
                'xgb': {
                    'eta': 0.04,
                    'max_depth': 7,
                    'n_estimators': 381,
                    'min_child_weight': 20,
                    'reg_alpha': 0.0,
                    'reg_lambda': 0.1,
                },
            },
            'diffusion': {
                'lgb': {
                    'num_leaves': 22,
                    'learning_rate': 0.14,
                    'max_depth': 11,
                    'n_estimators': 550,
                    'min_child_samples': 48,
                    'reg_alpha': 0.0,
                    'reg_lambda': 0.9,
                },
                'xgb': {
                    'eta': 0.2,
                    'max_depth': 10,
                    'n_estimators': 120,
                    'min_child_weight': 29,
                    'reg_alpha': 0.2,
                    'reg_lambda': 0.2,
                },
            },
        },
        'original': {
            'drift': {
                'lgb': {
                    'num_leaves': 78,
                    'learning_rate': 0.02,
                    'max_depth': 3,
                    'n_estimators': 465,
                    'min_child_samples': 76,
                    'reg_alpha': 0.1,
                    'reg_lambda': 0.9,
                },
                'xgb': {
                    'eta': 0.02,
                    'max_depth': 4,
                    'n_estimators': 762,
                    'min_child_weight': 28,
                    'reg_alpha': 0.2,
                    'reg_lambda': 0.1,
                },
            },
            'diffusion': {
                'lgb': {
                    'num_leaves': 22,
                    'learning_rate': 0.13,
                    'max_depth': 11,
                    'n_estimators': 550,
                    'min_child_samples': 48,
                    'reg_alpha': 0.0,
                    'reg_lambda': 0.9,
                },
                'xgb': {
                    'eta': 0.2,
                    'max_depth': 10,
                    'n_estimators': 120,
                    'min_child_weight': 29,
                    'reg_alpha': 0.2,
                    'reg_lambda': 0.2,
                },
            },
        }
    }
}

# model parameters for the v2 version of machine learning. Includes more models but only for detrended data
parameters_v2 = {
    'AUS': {
        'drift': {
            'gbt_lgb': {
                'num_leaves': 25,
                'learning_rate': 0.01,
                'max_depth': 10,
                'n_estimators': 610,
                'min_child_samples': 39,
                'reg_alpha': 0.4,
                'reg_lambda': 0.1
            },
            'gbt_xgb_squarederror': {
                'eta': 0.09,
                'max_depth': 6,
                'n_estimators': 950,
                'min_child_weight': 11,
                'reg_alpha': 0.5,
                'reg_lambda': 0.6
            },
            'gbt_xgb_absoluteerror': {
                'eta': 0.01,
                'max_depth': 9,
                'n_estimators': 530,
                'min_child_weight': 13,
                'reg_alpha': 0.06,
                'reg_lambda': 0.6
            },
            'rf_lgb': {
                'num_leaves': 51,
                'max_depth': 8,
                'n_estimators': 918,
                'min_child_samples': 31,
                'reg_alpha': 0.015,
                'reg_lambda': 0.3,
                'bagging_freq': 3,
                'bagging_fraction': 0.5
            },
            'mlp': {
                'activation': 'relu',
                'learning_rate_init': 0.001,
                'hidden_layer_sizes': (50, 50, 50, 50),
                'alpha': 0.01,
            }
        },
        'diffusion': {
            'gbt_lgb': {
                'num_leaves': 22,
                'learning_rate': 0.14,
                'max_depth': 11,
                'n_estimators': 550,
                'min_child_samples': 48,
                'reg_alpha': 0.0004,
                'reg_lambda': 0.9
            },
            'gbt_xgb_squarederror': {
                'eta': 0.42,
                'max_depth': 6,
                'n_estimators': 510,
                'min_child_weight': 24,
                'reg_alpha': 0.0,
                'reg_lambda': 0.3
            },
            'gbt_xgb_absoluteerror': {
                'eta': 0.05,
                'max_depth': 11,
                'n_estimators': 624,
                'min_child_weight': 17,
                'reg_alpha': 0.6,
                'reg_lambda': 0.7
            },
            'rf_lgb': {
                'num_leaves': 21,
                'max_depth': 6,
                'n_estimators': 737,
                'min_child_samples': 76,
                'reg_alpha': 0.001,
                'reg_lambda': 0.3,
                'bagging_freq': 6,
                'bagging_fraction': 0.3
            },
            'mlp': {
                'activation': 'identity',
                'learning_rate_init': 0.01,
                'hidden_layer_sizes': (10, 50, 50, 10),
                'alpha': 0.0001,
            }
        }
    },
    'CE': {
        'drift': {
            'gbt_lgb': {
                'num_leaves': 14,
                'learning_rate': 0.03,
                'max_depth': 5,
                'n_estimators': 873,
                'min_child_samples': 75,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0
            },
            'gbt_xgb_squarederror': {
                'eta': 0.04,
                'max_depth': 7,
                'n_estimators': 381,
                'min_child_weight': 20,
                'reg_alpha': 0.0,
                'reg_lambda': 0.1
            },
            'gbt_xgb_absoluteerror': {
                'eta': 0.02,
                'max_depth': 10,
                'n_estimators': 988,
                'min_child_weight': 16,
                'reg_alpha': 1.0,
                'reg_lambda': 0.8
            },
            'rf_lgb': {
                'num_leaves': 73,
                'max_depth': 10,
                'n_estimators': 669,
                'min_child_samples': 52,
                'reg_alpha': 0.007,
                'reg_lambda': 0.5,
                'bagging_freq': 6,
                'bagging_fraction': 0.4
            },
            'mlp': {
                'activation': 'relu',
                'learning_rate_init': 0.005,
                'hidden_layer_sizes': (20, 100, 100, 20),
                'alpha': 0.0005,
            }
        },
        'diffusion': {
            'gbt_lgb': {
                'num_leaves': 22,
                'learning_rate': 0.14,
                'max_depth': 11,
                'n_estimators': 550,
                'min_child_samples': 48,
                'reg_alpha': 0.0,
                'reg_lambda': 0.9
            },
            'gbt_xgb_squarederror': {
                'eta': 0.2,
                'max_depth': 10,
                'n_estimators': 120,
                'min_child_weight': 29,
                'reg_alpha': 0.2,
                'reg_lambda': 0.2
            },
            'gbt_xgb_absoluteerror': {
                'eta': 0.07,
                'max_depth': 11,
                'n_estimators': 900,
                'min_child_weight': 14,
                'reg_alpha': 0.6,
                'reg_lambda': 0.9
            },
            'rf_lgb': {
                'num_leaves': 21,
                'max_depth': 6,
                'n_estimators': 737,
                'min_child_samples': 76,
                'reg_alpha': 0.001,
                'reg_lambda': 0.3,
                'bagging_freq': 6,
                'bagging_fraction': 0.3
            },
            'mlp': {
                'activation': 'identity',
                'learning_rate_init': 0.005,
                'hidden_layer_sizes': (10, 100, 10),
                'alpha': 0.0001,
            }
        }
    }
}

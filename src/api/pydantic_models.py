from pydantic import BaseModel


class PredictionInput(BaseModel):

    # =========================
    # NUMERICAL FEATURES
    # =========================
    num__Total_Transaction_Amount: float
    num__Average_Transaction_Amount: float
    num__Transaction_Count: float
    num__Std_Transaction_Amount: float
    num__Max_Transaction_Amount: float
    num__Min_Transaction_Amount: float
    num__Total_Transaction_Value: float
    num__CountryCode: float
    num__PricingStrategy: float

    # =========================
    # CATEGORICAL (ONE-HOT ENCODED)
    # =========================

    cat__CurrencyCode_UGX: float

    cat__ProviderId_ProviderId_1: float
    cat__ProviderId_ProviderId_2: float
    cat__ProviderId_ProviderId_3: float
    cat__ProviderId_ProviderId_4: float
    cat__ProviderId_ProviderId_5: float
    cat__ProviderId_ProviderId_6: float

    cat__ProductCategory_airtime: float
    cat__ProductCategory_data_bundles: float
    cat__ProductCategory_financial_services: float
    cat__ProductCategory_movies: float
    cat__ProductCategory_other: float
    cat__ProductCategory_ticket: float
    cat__ProductCategory_transport: float
    cat__ProductCategory_tv: float
    cat__ProductCategory_utility_bill: float

    cat__ChannelId_ChannelId_1: float
    cat__ChannelId_ChannelId_2: float
    cat__ChannelId_ChannelId_3: float
    cat__ChannelId_ChannelId_5: float


class PredictionOutput(BaseModel):

    risk_probability: float
    risk_label: int

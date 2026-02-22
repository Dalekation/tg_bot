import numpy as np
import random
from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Tuple
from enum import Enum
import uvicorn
from dataclasses import dataclass
import time
import secrets
from itertools import product
import traceback
import csv
import os
from datetime import datetime
from pathlib import Path




# ========== КОНФИГУРАЦИЯ ==========

# Токены доступа и их параметры
API_TOKENS = {
    "take_on_me_take_me_on": {
        "name": "Руслан+Алена+Дарья+Ярослав+Илья",
        "requests_remaining": 80,
        "total_requests": 80,
        "budget": 4000000,  # Бюджет на скидки для этого токена
        "budget_remaining": 4000000,
        "settings": {
            "min_users_per_group": 8000,
            "max_users_per_group": 12000,
            "behavior_distribution": {
                "price_sensitive": 0.20,
                "loyal": 0.15,
                "hesitant": 0.25,
                "never_buyer": 0.40
            },
            "base_prices": {"provider1": 300, "provider2": 250},
            "reliability": {"provider1": 0.9, "provider2": 0.7}
        }
    },
    "never_gonna_give_you_up": {
        "name": "Никита-Никита-Никита-Анна-Анастасия",
        "requests_remaining": 80,
        "total_requests": 80,
        "budget": 4000000,  # Бюджет на скидки для этого токена
        "budget_remaining": 4000000,
        "settings": {
            "min_users_per_group": 8000,
            "max_users_per_group": 12000,
            "behavior_distribution": {
                "price_sensitive": 0.25,
                "loyal": 0.20,
                "hesitant": 0.30,
                "never_buyer": 0.25
            },
            "base_prices": {"provider1": 320, "provider2": 270},
            "reliability": {"provider1": 0.85, "provider2": 0.75}
        }
    },
    "admin_token_789": {
        "name": "Администратор",
        "requests_remaining": 1000,
        "budget_remaining": 50000,
        "total_requests": 1000,
        "budget": 50000,  # Бюджет на скидки для админа
        "settings": {
            "min_users_per_group": 100,
            "max_users_per_group": 1000,
            "behavior_distribution": {
                "price_sensitive": 0.20,
                "loyal": 0.15,
                "hesitant": 0.25,
                "never_buyer": 0.40
            },
            "base_prices": {"provider1": 300, "provider2": 250},
            "reliability": {"provider1": 0.9, "provider2": 0.6}
        }
    }
}

# Глобальные базовые цены (будут переопределяться настройками токена)
BASE_PRICES = {
    "provider1": 300,
    "provider2": 250
}

# Диапазоны для случайного количества пользователей (будут переопределяться настройками токена)
MIN_USERS_PER_GROUP = 8000
MAX_USERS_PER_GROUP = 10000


# ========== МОДЕЛИ ДАННЫХ ==========

class UserBehavior(str, Enum):
    PRICE_SENSITIVE = "price_sensitive"
    LOYAL = "loyal"
    HESITANT = "hesitant"
    NEVER_BUYER = "never_buyer"

class GroupDiscounts(BaseModel):
    """Скидки для группы"""
    discount1: float = Field(..., ge=0, le=1, description="Скидка на поставщика 1 (0-1)")
    discount2: float = Field(..., ge=0, le=1, description="Скидка на поставщика 2 (0-1)")

class SimulationRequest(BaseModel):
    """Запрос на симуляцию"""
    group1: GroupDiscounts
    group2: GroupDiscounts
    seed: Optional[int] = Field(None, description="Seed для воспроизводимости результатов")

class GroupStats(BaseModel):
    """Статистика по группе"""
    discount1: float
    discount2: float
    user_count: int
    no_order: int
    provider1: int
    provider2: int

class SimulationResponse(BaseModel):
    """Упрощённый ответ на симуляцию"""
    group1: GroupStats
    group2: GroupStats

class BasePricesResponse(BaseModel):
    """Ответ с базовыми ценами"""
    provider1_price: float
    provider2_price: float

class TokenSettingsResponse(BaseModel):
    """Настройки токена"""
    name: str
    requests_remaining: int
    total_requests: int
    budget: float
    settings: Dict

class TokenInfoResponse(BaseModel):
    """Информация о токене"""
    token_name: str
    requests_remaining: int
    total_requests: int
    budget: float
    is_valid: bool
    settings: Dict

class TokenCreateRequest(BaseModel):
    """Запрос на создание токена"""
    name: str = Field(..., description="Название для токена")
    requests_limit: int = Field(80, ge=1, le=10000, description="Лимит запросов")
    budget: float = Field(5000, ge=0, description="Бюджет на скидки")
    settings: Optional[Dict] = Field(None, description="Настройки для токена")

class TokenUpdateSettingsRequest(BaseModel):
    """Запрос на обновление настроек токена"""
    min_users_per_group: Optional[int] = Field(None, ge=10, le=10000)
    max_users_per_group: Optional[int] = Field(None, ge=10, le=10000)
    behavior_distribution: Optional[Dict[str, float]] = None
    base_prices: Optional[Dict[str, float]] = None
    reliability: Optional[Dict[str, float]] = None
    budget: Optional[float] = Field(None, ge=0, description="Новый бюджет на скидки")

class TokenBudgetUpdateRequest(BaseModel):
    """Запрос на обновление бюджета токена"""
    budget: float = Field(..., ge=0, description="Новый бюджет на скидки")

class BudgetOptimizationRequest(BaseModel):
    """Запрос на оптимизацию скидок с учётом бюджета"""
    target_token: str = Field(..., description="Токен, для которого выполняется оптимизация")
    group1_users: Optional[int] = Field(None, description="Количество пользователей в группе 1 (если не указано - случайное)")
    group2_users: Optional[int] = Field(None, description="Количество пользователей в группе 2 (если не указано - случайное)")
    discount_step: float = Field(0.05, ge=0.01, le=0.2, description="Шаг перебора скидок")
    max_discount: float = Field(0.5, ge=0.1, le=1.0, description="Максимальная скидка")
    optimization_metric: str = Field("orders", description="Метрика для оптимизации: orders, provider1, provider2, no_order, revenue")
    iterations: int = Field(3, ge=1, le=10, description="Количество итераций для усреднения")
    use_token_budget: bool = Field(True, description="Использовать бюджет из настроек токена")
    custom_budget: Optional[float] = Field(None, ge=0, description="Свой бюджет (если use_token_budget=False)")
    seed: Optional[int] = Field(None, description="Seed для воспроизводимости")

class BudgetOptimizationResult(BaseModel):
    """Результат оптимизации с учётом бюджета"""
    target_token: str
    token_name: str
    best_discounts_group1: Tuple[float, float]
    best_discounts_group2: Tuple[float, float]
    best_metric_value: float
    total_discount_cost: float
    budget_used: float
    budget_limit: float
    budget_usage_percent: float
    metric: str
    total_orders: int
    total_revenue: float
    base_orders: int
    orders_increase: int
    orders_increase_percent: float
    roi: float  # Return on Investment (прирост заказов / затраты на скидки * 1000)
    top_5_scenarios: List[Dict]
    settings_used: Dict

class OptimizationScenario(BaseModel):
    """Сценарий оптимизации для сравнения"""
    name: str
    discounts_group1: Tuple[float, float]
    discounts_group2: Tuple[float, float]
    total_orders: int
    total_revenue: float
    discount_cost: float
    roi: float

# ========== НАСТРОЙКИ ЛОГИРОВАНИЯ ==========

LOG_DIR = "simulation_logs"
LOG_FILE = os.path.join(LOG_DIR, "simulation_history.csv")

# Создаём директорию для логов если её нет
os.makedirs(LOG_DIR, exist_ok=True)

# Создаём файл с заголовками если его нет
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'timestamp', 'token_name', 'token_masked',
            'discount1_g1', 'discount2_g1', 'user_count1',
            'discount1_g2', 'discount2_g2', 'user_count2',
            'seed', 'no_order_total', 'provider1_total', 'provider2_total',
            'no_order_g1', 'provider1_g1', 'provider2_g1',
            'no_order_g2', 'provider1_g2', 'provider2_g2'
        ])

def log_simulation_to_csv(token_info: Dict, request: SimulationRequest, 
                          results1: Dict, results2: Dict, 
                          user_count1: int, user_count2: int):
    """
    Записывает результаты симуляции в CSV файл
    """
    try:
        # Маскируем токен (показываем только первые 8 символов)
        token_masked = request.headers.get("authorization", "").replace("Bearer ", "")[:8] + "..." if hasattr(request, 'headers') else "unknown"
        
        # Подготавливаем данные для записи
        row = [
            datetime.now().isoformat(),              # timestamp
            token_info.get("name", "unknown"),       # token_name
            token_masked,                            # token_masked
            request.group1.discount1,                 # discount1_g1
            request.group1.discount2,                 # discount2_g1
            user_count1,                              # user_count1
            request.group2.discount1,                 # discount1_g2
            request.group2.discount2,                 # discount2_g2
            user_count2,                              # user_count2
            request.seed if request.seed else "None", # seed
            results1['no_order'] + results2['no_order'],  # no_order_total
            results1['provider1'] + results2['provider1'], # provider1_total
            results1['provider2'] + results2['provider2'], # provider2_total
            results1['no_order'],                      # no_order_g1
            results1['provider1'],                      # provider1_g1
            results1['provider2'],                      # provider2_g1
            results2['no_order'],                      # no_order_g2
            results1['provider1'],                      # provider1_g2 (исправлено: было provider1_g1)
            results2['provider2']                       # provider2_g2
        ]
        
        # Исправляем опечатку в индексах
        row[17] = results2['provider1']  # provider1_g2
        row[18] = results2['provider2']  # provider2_g2
        
        # Записываем в CSV
        with open(LOG_FILE, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(row)
            
    except Exception as e:
        print(f"Ошибка при записи в лог: {e}")

# ========== АУТЕНТИФИКАЦИЯ ==========

security = HTTPBearer()
token_store = API_TOKENS.copy()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Проверяет токен доступа и возвращает информацию о нём"""
    token = credentials.credentials
    
    if token not in token_store:
        raise HTTPException(
            status_code=401,
            detail="Недействительный токен",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    token_info = token_store[token]
    
    if token_info["requests_remaining"] <= 0:
        raise HTTPException(
            status_code=429,
            detail="Превышен лимит запросов для данного токена",
        )
    
    return token, token_info

def get_token_settings(token: str) -> Dict:
    """Возвращает настройки для токена"""
    if token in token_store:
        return token_store[token].get("settings", {})
    return {}

def get_token_budget(token: str) -> float:
    """Возвращает бюджет для токена"""
    if token in token_store:
        return token_store[token].get("budget", 5000)
    return 5000

def decrement_token_usage(token: str):
    """Уменьшает счётчик запросов для токена"""
    if token in token_store:
        token_store[token]["requests_remaining"] -= 1

def decrement_token_budget(token: str, budget: int):
    """Уменьшает счётчик запросов для токена"""
    if token in token_store:
        token_store[token]["budget_remaining"] -= budget

def generate_token(name: str, requests_limit: int, budget: float, settings: Dict = None) -> str:
    """Генерирует новый токен"""
    random_part = secrets.token_urlsafe(32)
    token = f"{name}_{random_part}"
    
    default_settings = {
        "min_users_per_group": 100,
        "max_users_per_group": 1000,
        "behavior_distribution": {
            "price_sensitive": 0.20,
            "loyal": 0.15,
            "hesitant": 0.25,
            "never_buyer": 0.40
        },
        "base_prices": {"provider1": 300, "provider2": 250},
        "reliability": {"provider1": 0.9, "provider2": 0.6}
    }
    
    if settings:
        default_settings.update(settings)
    
    token_store[token] = {
        "name": name,
        "requests_remaining": requests_limit,
        "total_requests": requests_limit,
        "budget": budget,
        "settings": default_settings
    }
    
    return token


# ========== МОДЕЛЬ СИМУЛЯТОРА ==========

@dataclass
class HyperParameters:
    """Гиперпараметры модели"""
    base_price_provider1: float = 300
    base_price_provider2: float = 250
    reliability_provider1: float = 0.9
    reliability_provider2: float = 0.6
    price_sensitive_intent_range: Tuple = (0.3, 0.6)
    loyal_intent_range: Tuple = (0.2, 0.5)
    loyal_switch_threshold_range: Tuple = (0.2, 0.4)
    loyal_strength_range: Tuple = (0.6, 0.9)
    hesitant_intent_range: Tuple = (0.1, 0.3)
    hesitant_discount_sensitivity_range: Tuple = (0.5, 1.0)
    hesitant_price_weight_range: Tuple = (0.4, 0.8)
    hesitant_quality_weight_range: Tuple = (0.2, 0.4)
    max_order_prob: float = 0.8
    max_hesitant_order_prob: float = 0.7
    discount_boost_factor: float = 0.3


class User:
    """Класс пользователя"""
    def __init__(self, user_id: int, behavior: UserBehavior, hp: HyperParameters):
        self.id = user_id
        # Явно преобразуем в обычную строку Python
        self.behavior = str(behavior.value)
        
        if behavior == UserBehavior.PRICE_SENSITIVE:
            self.base_intent = float(np.random.uniform(*hp.price_sensitive_intent_range))
            self.price_weight = 1.0
            self.switch_threshold = 0.0
            
        elif behavior == UserBehavior.LOYAL:
            self.preferred = int(np.random.choice([1, 2]))
            self.base_intent = float(np.random.uniform(*hp.loyal_intent_range))
            self.switch_threshold = float(np.random.uniform(*hp.loyal_switch_threshold_range))
            self.loyalty_strength = float(np.random.uniform(*hp.loyal_strength_range))
            
        elif behavior == UserBehavior.HESITANT:
            self.base_intent = float(np.random.uniform(*hp.hesitant_intent_range))
            self.discount_sensitivity = float(np.random.uniform(*hp.hesitant_discount_sensitivity_range))
            self.price_weight = float(np.random.uniform(*hp.hesitant_price_weight_range))
            self.quality_weight = float(np.random.uniform(*hp.hesitant_quality_weight_range))
            
        elif behavior == UserBehavior.NEVER_BUYER:
            self.base_intent = 0.0
            self.never_buy = True


class DeliverySimulator:
    """Симулятор доставки"""
    
    def __init__(self, hp: HyperParameters = None):
        self.hp = hp or HyperParameters()
        self.users = []
        
    def add_users(self, count: int, behavior_distribution: Dict[UserBehavior, float]):
        """Добавляет пользователей с заданным распределением поведения"""
        start_id = len(self.users)
        
        for behavior, proportion in behavior_distribution.items():
            n_type = int(count * proportion)
            for i in range(n_type):
                user = User(len(self.users), behavior, self.hp)
                self.users.append(user)
        
        # Добавляем недостающих пользователей
        while len(self.users) - start_id < count:
            behaviors = list(behavior_distribution.keys())
            probs = list(behavior_distribution.values())
            # Используем random.choice вместо numpy.random.choice для избежания numpy типов
            behavior = random.choices(behaviors, weights=probs)[0]
            user = User(len(self.users), behavior, self.hp)
            self.users.append(user)
    
    def clear_users(self):
        """Очищает список пользователей"""
        self.users = []
    
    def simulate_with_financials(self, discount1: float, discount2: float):
        """
        Запускает симуляцию для текущего набора пользователей с финансовыми показателями
        """
        hp = self.hp
        
        # Финальные цены со скидками
        price1 = float(hp.base_price_provider1 * (1 - discount1))
        price2 = float(hp.base_price_provider2 * (1 - discount2))
        
        max_discount = float(max(discount1, discount2))
        
        results = {
            'no_order': 0,
            'provider1': 0,
            'provider2': 0,
            'total_users': len(self.users),
            'total_paid': 0.0,
            'total_discount_cost': 0.0,
            'revenue_without_discounts': 0.0
        }
        
        for user in self.users:
            # Определяем, будет ли заказывать
            will_order = self._will_order(user, max_discount)
            
            if not will_order or (hasattr(user, 'never_buy') and user.never_buy):
                results['no_order'] += 1
            else:
                choice = self._choose_provider(user, price1, price2, 
                                              hp.reliability_provider1, hp.reliability_provider2,
                                              discount1, discount2, 
                                              hp.base_price_provider1, hp.base_price_provider2)
                
                if choice == 1:
                    results['provider1'] += 1
                    results['total_paid'] += price1
                    results['total_discount_cost'] += float(hp.base_price_provider1 - price1)
                    results['revenue_without_discounts'] += float(hp.base_price_provider1)
                elif choice == 2:
                    results['provider2'] += 1
                    results['total_paid'] += price2
                    results['total_discount_cost'] += float(hp.base_price_provider2 - price2)
                    results['revenue_without_discounts'] += float(hp.base_price_provider2)
                else:
                    results['no_order'] += 1
        
        results['total_orders'] = results['provider1'] + results['provider2']
        results['total_users'] = int(results['total_users'])

        
        
        return results
    
    def _will_order(self, user, max_discount):
        """Определяет, будет ли пользователь заказывать"""
        if hasattr(user, 'never_buy') and user.never_buy:
            return False
        
        base_prob = float(user.base_intent)
        
        # Сравниваем со строковыми значениями Enum
        if user.behavior == UserBehavior.HESITANT.value:
            discount_boost = float(max_discount * user.discount_sensitivity)
            prob = min(base_prob + discount_boost, float(self.hp.max_hesitant_order_prob))
        else:
            prob = min(base_prob + float(max_discount * self.hp.discount_boost_factor), 
                      float(self.hp.max_order_prob))
        
        return random.random() < prob
    
    def _choose_provider(self, user, price1, price2, rel1, rel2, discount1, discount2, base_price1, base_price2):
        """Выбирает поставщика"""
        
        # Сравниваем со строковыми значениями Enum
        if user.behavior == UserBehavior.PRICE_SENSITIVE.value:
            if price1 < price2:
                return 1
            elif price2 < price1:
                return 2
            else:
                return random.choice([1, 2])
        
        elif user.behavior == UserBehavior.LOYAL.value:
            preferred = int(user.preferred)
            other = 3 - preferred
            
            price_pref = float(price1 if preferred == 1 else price2)
            price_other = float(price2 if preferred == 1 else price1)
            
            if price_pref > 0:
                price_diff = (price_pref - price_other) / price_pref
            else:
                price_diff = 0
            
            if price_diff > float(user.switch_threshold):
                return other
            else:
                return preferred
        
        elif user.behavior == UserBehavior.HESITANT.value:
            saving1 = float((base_price1 - price1) / base_price1) if base_price1 > 0 else 0
            saving2 = float((base_price2 - price2) / base_price2) if base_price2 > 0 else 0
            
            attractiveness1 = saving1 * 0.7 + float(rel1) * 0.3
            attractiveness2 = saving2 * 0.7 + float(rel2) * 0.3
            
            attractiveness1 += float(np.random.normal(0, 0.1))
            attractiveness2 += float(np.random.normal(0, 0.1))
            
            if attractiveness1 > attractiveness2 + 0.1:
                return 1
            elif attractiveness2 > attractiveness1 + 0.1:
                return 2
            else:
                return random.choice([1, 2])
        
        return 0


# ========== FastAPI ПРИЛОЖЕНИЕ ==========

app = FastAPI(
    title="Delivery Discount Simulator API",
    description="API для моделирования поведения пользователей при выборе доставки со скидками",
    version="1.0.0"
)


# ========== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ==========

def get_simulator_with_token_settings(token: str) -> Tuple[DeliverySimulator, Dict]:
    """Создаёт симулятор с настройками токена"""
    settings = get_token_settings(token)
    
    # Явно преобразуем все значения в float
    hp = HyperParameters(
        base_price_provider1=float(settings.get("base_prices", {}).get("provider1", 300)),
        base_price_provider2=float(settings.get("base_prices", {}).get("provider2", 250)),
        reliability_provider1=float(settings.get("reliability", {}).get("provider1", 0.9)),
        reliability_provider2=float(settings.get("reliability", {}).get("provider2", 0.6))
    )
    
    return DeliverySimulator(hp), settings


def calculate_roi(base_orders: int, new_orders: int, discount_cost: float) -> float:
    """Рассчитывает ROI (окупаемость инвестиций в скидки) - заказов на 1000 ден. единиц"""
    if discount_cost == 0:
        return 0
    
    order_increase = new_orders - base_orders
    roi = order_increase / discount_cost * 1000  # ROI в заказах на 1000 ден. единиц
    
    return roi


# ========== ЭНДПОИНТЫ ==========

@app.get("/")
async def root():
    """Корневой эндпоинт"""
    return {
        "message": "Delivery Discount Simulator API",
        "docs": "/docs",
        "endpoints": {
            "POST /simulate": "Запуск симуляции для двух групп",
            # "POST /optimize/budget": "Оптимизация скидок с учётом бюджета для указанного токена",
            "GET /prices": "Получение базовых цен",
            "GET /token/info": "Информация о текущем токене"
            # "GET /token/settings": "Настройки текущего токена",
            # "POST /token/create": "Создание нового токена (admin only)",
            # "PUT /token/settings": "Обновление настроек токена (admin only)",
            # "PUT /token/budget": "Обновление бюджета токена (admin only)"
        }
    }


@app.get("/health")
async def health():
    """Проверка состояния API"""
    return {"status": "healthy", "timestamp": time.time()}


@app.get("/prices", response_model=BasePricesResponse)
async def get_base_prices(
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """
    Возвращает базовые цены поставщиков для текущего токена
    """
    token, token_info = verify_token(credentials)
    settings = get_token_settings(token)
    
    return BasePricesResponse(
        provider1_price=settings.get("base_prices", {}).get("provider1", 300),
        provider2_price=settings.get("base_prices", {}).get("provider2", 250)
    )


@app.get("/token/info", response_model=TokenInfoResponse)
async def get_token_info(
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """
    Возвращает информацию о текущем токене
    """
    token, token_info = verify_token(credentials)
    
    return TokenInfoResponse(
        token_name=token_info["name"],
        requests_remaining=token_info["requests_remaining"],
        total_requests=token_info["total_requests"],
        budget=token_info.get("budget_remaining", 5000),
        is_valid=True
        # settings=token_info.get("settings", {})
    )


@app.get("/token/settings", response_model=TokenSettingsResponse)
async def get_token_settings_endpoint(
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """
    Возвращает настройки текущего токена
    """
    token, token_info = verify_token(credentials)
    
    return TokenSettingsResponse(
        name=token_info["name"],
        requests_remaining=token_info["requests_remaining"],
        total_requests=token_info["total_requests"],
        budget=token_info.get("budget_remaining", 5000)
        # settings=token_info.get("settings", {})
    )


@app.post("/token/create")
async def create_token(
    request: TokenCreateRequest,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """
    Создаёт новый токен (только для администратора)
    """
    token, token_info = verify_token(credentials)
    
    # Проверяем, что это админский токен
    if token != "admin_token_789" and token_info["name"] != "Администратор":
        raise HTTPException(
            status_code=403,
            detail="Недостаточно прав для создания токенов"
        )
    
    new_token = generate_token(request.name, request.requests_limit, request.budget, request.settings)
    
    return {
        "token": new_token,
        "name": request.name,
        "requests_limit": request.requests_limit,
        "budget": request.budget,
        "settings": request.settings,
        "message": "Токен успешно создан. Сохраните его, он больше не будет показан."
    }


@app.put("/token/settings")
async def update_token_settings(
    settings: TokenUpdateSettingsRequest,
    target_token: str = Query(..., description="Токен, для которого меняются настройки"),
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """
    Обновляет настройки для указанного токена (только для администратора)
    """
    token, token_info = verify_token(credentials)
    
    # Проверяем, что это админский токен
    if token != "admin_token_789" and token_info["name"] != "Администратор":
        raise HTTPException(
            status_code=403,
            detail="Недостаточно прав для изменения настроек"
        )
    
    if target_token not in token_store:
        raise HTTPException(status_code=404, detail="Токен не найден")
    
    current_settings = token_store[target_token].get("settings", {})
    
    # Обновляем настройки
    if settings.min_users_per_group is not None:
        current_settings["min_users_per_group"] = settings.min_users_per_group
    if settings.max_users_per_group is not None:
        current_settings["max_users_per_group"] = settings.max_users_per_group
    if settings.behavior_distribution is not None:
        current_settings["behavior_distribution"] = settings.behavior_distribution
    if settings.base_prices is not None:
        if "base_prices" not in current_settings:
            current_settings["base_prices"] = {}
        current_settings["base_prices"].update(settings.base_prices)
    if settings.reliability is not None:
        if "reliability" not in current_settings:
            current_settings["reliability"] = {}
        current_settings["reliability"].update(settings.reliability)
    if settings.budget is not None:
        token_store[target_token]["budget"] = settings.budget
    
    token_store[target_token]["settings"] = current_settings
    
    return {
        "message": "Настройки обновлены",
        "token": target_token[:8] + "...",
        "budget": token_store[target_token].get("budget"),
        "settings": current_settings
    }


@app.put("/token/budget")
async def update_token_budget(
    request: TokenBudgetUpdateRequest,
    target_token: str = Query(..., description="Токен, для которого меняется бюджет"),
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """
    Обновляет бюджет для указанного токена (только для администратора)
    """
    token, token_info = verify_token(credentials)
    
    # Проверяем, что это админский токен
    if token != "admin_token_789" and token_info["name"] != "Администратор":
        raise HTTPException(
            status_code=403,
            detail="Недостаточно прав для изменения бюджета"
        )
    
    if target_token not in token_store:
        raise HTTPException(status_code=404, detail="Токен не найден")
    
    token_store[target_token]["budget"] = request.budget
    token_store[target_token]["budget_remaining"] = request.budget
    
    return {
        "message": "Бюджет обновлён",
        "token": target_token[:8] + "...",
        "new_budget": request.budget
    }


@app.post("/simulate", response_model=SimulationResponse)
async def simulate(
    request: SimulationRequest,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """
    Запускает симуляцию для двух групп пользователей с разными наборами скидок
    """
    token, token_info = verify_token(credentials)
    
    try:
        # Уменьшаем счётчик запросов
        decrement_token_usage(token)

        budget_left = token_info.get("budget_remaining", 0)

        if budget_left <= 0:
            raise HTTPException(status_code=400, detail="Денег нет, но вы держитесь")
        

        # Получаем настройки токена
        settings = token_info.get("settings", {})
        min_users = int(settings.get("min_users_per_group", 8000))
        max_users = int(settings.get("max_users_per_group", 10000))

        base_prices = settings.get("base_prices", {})
        price1 = int(base_prices.get("min_users_per_group", 300))
        price2 = int(base_prices.get("max_users_per_group", 250))


        behavior_dist_raw = settings.get("behavior_distribution", {
            "price_sensitive": 0.20,
            "loyal": 0.15,
            "hesitant": 0.25,
            "never_buyer": 0.40
        })
        
        # Преобразуем распределение поведения в формат Enum с float значениями
        behavior_dist = {
            UserBehavior.PRICE_SENSITIVE: float(behavior_dist_raw.get("price_sensitive", 0.20)),
            UserBehavior.LOYAL: float(behavior_dist_raw.get("loyal", 0.15)),
            UserBehavior.HESITANT: float(behavior_dist_raw.get("hesitant", 0.25)),
            UserBehavior.NEVER_BUYER: float(behavior_dist_raw.get("never_buyer", 0.40))
        }
        
        # Нормализуем сумму до 1
        total = sum(behavior_dist.values())
        if total != 1.0:
            for key in behavior_dist:
                behavior_dist[key] /= total
        
        # Устанавливаем seed если указан
        # if request.seed is not None:
        #     random.seed(request.seed)
        #     np.random.seed(request.seed)
        
        # Генерируем случайное количество пользователей для каждой группы
        user_count1 = random.randint(min_users, max_users)
        user_count2 = random.randint(min_users, max_users)
        
        # Создаем симулятор с настройками токена
        sim, _ = get_simulator_with_token_settings(token)
        
        # Симуляция для первой группы
        sim.add_users(user_count1, behavior_dist)
        results1 = sim.simulate_with_financials(
            float(request.group1.discount1), 
            float(request.group1.discount2)
        )
        
        # Симуляция для второй группы
        sim.clear_users()
        sim.add_users(user_count2, behavior_dist)
        results2 = sim.simulate_with_financials(
            float(request.group2.discount1), 
            float(request.group2.discount2)
        )

        # Логируем результаты в CSV
        try:
            log_simulation_to_csv(
                token_info, request, results1, results2, 
                user_count1, user_count2
            )
        except Exception as log_error:
            # Логируем ошибку, но не прерываем выполнение
            print(f"Ошибка логирования: {log_error}")
        
        # Формируем ответ с явным преобразованием типов
        response = SimulationResponse(
            group1=GroupStats(
                discount1=float(request.group1.discount1),
                discount2=float(request.group1.discount2),
                user_count=int(results1['total_users']),
                no_order=int(results1['no_order']),
                provider1=int(results1['provider1']),
                provider2=int(results1['provider2'])
            ),
            group2=GroupStats(
                discount1=float(request.group2.discount1),
                discount2=float(request.group2.discount2),
                user_count=int(results2['total_users']),
                no_order=int(results2['no_order']),
                provider1=int(results2['provider1']),
                provider2=int(results2['provider2'])
            )
        )

        total_cost = \
            int(results1['provider1']) * float(request.group1.discount1) * price1 + \
            int(results1['provider2']) * float(request.group1.discount2) * price2 + \
            int(results2['provider1']) * float(request.group2.discount1) * price1 + \
            int(results2['provider2']) * float(request.group2.discount2) * price2
        
        decrement_token_budget(token, int(total_cost))
        
        return response
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/optimize/budget", response_model=BudgetOptimizationResult)
async def optimize_with_budget(
    request: BudgetOptimizationRequest,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """
    Находит оптимальные скидки с учётом ограниченного бюджета для указанного токена
    
    - **target_token**: токен, для которого выполняется оптимизация (используются его настройки)
    - **use_token_budget**: если True, используется бюджет из настроек токена, иначе из custom_budget
    - **custom_budget**: свой бюджет (если use_token_budget=False)
    - **optimization_metric**: метрика для оптимизации (orders, revenue, provider1, provider2)
    """
    admin_token, admin_info = verify_token(credentials)
    
    # Проверяем, что это админский токен
    if admin_token != "admin_token_789" and admin_info["name"] != "Администратор":
        raise HTTPException(
            status_code=403,
            detail="Недостаточно прав для оптимизации"
        )
    
    # Проверяем существование целевого токена
    if request.target_token not in token_store:
        raise HTTPException(status_code=404, detail="Целевой токен не найден")
    
    target_token_info = token_store[request.target_token]
    
    try:
        # Уменьшаем счётчик запросов админа
        decrement_token_usage(admin_token)
        
        # Определяем бюджет
        if request.use_token_budget:
            budget_limit = target_token_info.get("budget", 5000)
        else:
            if request.custom_budget is None:
                raise HTTPException(status_code=400, detail="Не указан custom_budget при use_token_budget=False")
            budget_limit = request.custom_budget
        
        # Получаем настройки целевого токена
        settings = target_token_info.get("settings", {})
        min_users = settings.get("min_users_per_group", 100)
        max_users = settings.get("max_users_per_group", 1000)
        behavior_dist_raw = settings.get("behavior_distribution", {
            "price_sensitive": 0.20,
            "loyal": 0.15,
            "hesitant": 0.25,
            "never_buyer": 0.40
        })
        
        behavior_dist = {
            UserBehavior.PRICE_SENSITIVE: behavior_dist_raw.get("price_sensitive", 0.20),
            UserBehavior.LOYAL: behavior_dist_raw.get("loyal", 0.15),
            UserBehavior.HESITANT: behavior_dist_raw.get("hesitant", 0.25),
            UserBehavior.NEVER_BUYER: behavior_dist_raw.get("never_buyer", 0.40)
        }
        
        # Устанавливаем seed
        # if request.seed is not None:
        #     np.random.seed(request.seed)
        #     random.seed(request.seed)
        
        # Генерируем или используем заданное количество пользователей
        user_count1 = request.group1_users if request.group1_users else random.randint(min_users, max_users)
        user_count2 = request.group2_users if request.group2_users else random.randint(min_users, max_users)
        
        # Создаём симулятор с настройками целевого токена
        sim, _ = get_simulator_with_token_settings(request.target_token)
        
        # Сначала симулируем базовый сценарий (без скидок)
        sim.add_users(user_count1, behavior_dist)
        base_results1 = sim.simulate_with_financials(0, 0)
        
        sim.clear_users()
        sim.add_users(user_count2, behavior_dist)
        base_results2 = sim.simulate_with_financials(0, 0)
        
        base_total_orders = base_results1['total_orders'] + base_results2['total_orders']
        base_total_revenue = base_results1['total_paid'] + base_results2['total_paid']
        
        # Создаём дискретные значения скидок для перебора
        discount_values = np.arange(0, request.max_discount + request.discount_step, request.discount_step)
        
        best_metric_value = -float('inf')
        best_discounts_g1 = (0, 0)
        best_discounts_g2 = (0, 0)
        best_total_discount_cost = 0
        best_total_orders = 0
        best_total_revenue = 0
        
        all_results = []
        
        # Перебираем все комбинации скидок
        total_combinations = len(discount_values) ** 4
        print(f"Перебор {total_combinations} комбинаций скидок для токена {request.target_token[:8]}...")
        
        for d1_g1, d2_g1, d1_g2, d2_g2 in product(discount_values, repeat=4):
            # Усредняем результаты по нескольким итерациям
            avg_results = {
                'total_orders': 0,
                'total_paid': 0,
                'total_discount_cost': 0,
                'provider1': 0,
                'provider2': 0
            }
            
            for iteration in range(request.iterations):
                # Симуляция для группы 1
                sim.clear_users()
                sim.add_users(user_count1, behavior_dist)
                res1 = sim.simulate_with_financials(d1_g1, d2_g1)
                
                # Симуляция для группы 2
                sim.clear_users()
                sim.add_users(user_count2, behavior_dist)
                res2 = sim.simulate_with_financials(d1_g2, d2_g2)
                
                avg_results['total_orders'] += (res1['total_orders'] + res2['total_orders']) / request.iterations
                avg_results['total_paid'] += (res1['total_paid'] + res2['total_paid']) / request.iterations
                avg_results['total_discount_cost'] += (res1['total_discount_cost'] + res2['total_discount_cost']) / request.iterations
                avg_results['provider1'] += (res1['provider1'] + res2['provider1']) / request.iterations
                avg_results['provider2'] += (res1['provider2'] + res2['provider2']) / request.iterations
            
            # Проверяем бюджетное ограничение
            if avg_results['total_discount_cost'] > budget_limit:
                continue
            
            # Вычисляем метрику для оптимизации
            if request.optimization_metric == "orders":
                metric_value = avg_results['total_orders']
            elif request.optimization_metric == "revenue":
                metric_value = avg_results['total_paid']
            elif request.optimization_metric == "provider1":
                metric_value = avg_results['provider1']
            elif request.optimization_metric == "provider2":
                metric_value = avg_results['provider2']
            elif request.optimization_metric == "no_order":
                metric_value = -avg_results['total_orders']  # Минимизируем не заказавших
            else:
                metric_value = avg_results['total_orders']
            
            # Сохраняем результаты
            roi = calculate_roi(
                base_total_orders,
                avg_results['total_orders'],
                avg_results['total_discount_cost']
            )
            
            result_entry = {
                'd1_g1': float(d1_g1),
                'd2_g1': float(d2_g1),
                'd1_g2': float(d1_g2),
                'd2_g2': float(d2_g2),
                'metric_value': float(metric_value),
                'total_orders': float(avg_results['total_orders']),
                'total_revenue': float(avg_results['total_paid']),
                'discount_cost': float(avg_results['total_discount_cost']),
                'roi': float(roi)
            }
            all_results.append(result_entry)
            
            # Обновляем лучшее решение
            if metric_value > best_metric_value:
                best_metric_value = metric_value
                best_discounts_g1 = (float(d1_g1), float(d2_g1))
                best_discounts_g2 = (float(d1_g2), float(d2_g2))
                best_total_discount_cost = float(avg_results['total_discount_cost'])
                best_total_orders = float(avg_results['total_orders'])
                best_total_revenue = float(avg_results['total_paid'])
        
        # Сортируем результаты по метрике и берём топ-5
        all_results.sort(key=lambda x: x['metric_value'], reverse=True)
        top_5 = all_results[:5]
        
        # Рассчитываем прирост заказов
        orders_increase = best_total_orders - base_total_orders
        orders_increase_percent = (orders_increase / base_total_orders * 100) if base_total_orders > 0 else 0
        
        # ROI для лучшего решения
        best_roi = calculate_roi(base_total_orders, best_total_orders, best_total_discount_cost)
        
        # Формируем ответ
        response = BudgetOptimizationResult(
            target_token=request.target_token[:8] + "...",
            token_name=target_token_info["name"],
            best_discounts_group1=best_discounts_g1,
            best_discounts_group2=best_discounts_g2,
            best_metric_value=best_metric_value,
            total_discount_cost=best_total_discount_cost,
            budget_used=best_total_discount_cost,
            budget_limit=budget_limit,
            budget_usage_percent=(best_total_discount_cost / budget_limit * 100) if budget_limit > 0 else 0,
            metric=request.optimization_metric,
            total_orders=int(best_total_orders),
            total_revenue=best_total_revenue,
            base_orders=base_total_orders,
            orders_increase=int(orders_increase),
            orders_increase_percent=orders_increase_percent,
            roi=best_roi,
            top_5_scenarios=top_5,
            settings_used={
                "user_count_group1": user_count1,
                "user_count_group2": user_count2,
                "min_users": min_users,
                "max_users": max_users,
                "behavior_distribution": behavior_dist_raw,
                "base_prices": settings.get("base_prices"),
                "iterations": request.iterations,
                "discount_step": request.discount_step
            }
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/admin/tokens")
async def list_tokens(
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """
    Список всех токенов с их бюджетами и настройками (только для администратора)
    """
    token, token_info = verify_token(credentials)
    
    if token != "admin_token_789" and token_info["name"] != "Администратор":
        raise HTTPException(
            status_code=403,
            detail="Недостаточно прав"
        )
    
    tokens_info = []
    for t, info in token_store.items():
        masked_token = t[:8] + "..." if len(t) > 8 else "***"
        tokens_info.append({
            "token": masked_token,
            "name": info["name"],
            "requests_remaining": info["requests_remaining"],
            "total_requests": info["total_requests"],
            "budget": info.get("budget", 5000),
            "usage_percent": (info["total_requests"] - info["requests_remaining"]) / info["total_requests"] * 100,
            "settings": info.get("settings", {})
        })
    
    return {"tokens": tokens_info}


# ========== ЗАПУСК ==========

if __name__ == "__main__":
    print("=" * 70)
    print("Delivery Discount Simulator API")
    print("=" * 70)
    print("\nДоступные токены для тестирования:")
    for token, info in API_TOKENS.items():
        print(f"  {token}  - {info['name']} (бюджет: {info['budget']}, осталось: {info['requests_remaining']} запросов)")
    
    print(f"\nКоличество пользователей в группах: случайно от {MIN_USERS_PER_GROUP} до {MAX_USERS_PER_GROUP}")
    print("\nЗапуск сервера на http://localhost:8000")
    print("Документация: http://localhost:8000/docs")
    print("=" * 70)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)

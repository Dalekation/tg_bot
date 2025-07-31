import logging
import sqlite3
from datetime import datetime, timedelta
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.utils.keyboard import InlineKeyboardBuilder
from aiogram.types import Message, CallbackQuery
import asyncio
from dotenv import dotenv_values

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Получение токена из окружения
config = dotenv_values(".env")

# Инициализация бота
bot = Bot(token=config['BOT_TOKEN'])
dp = Dispatcher()

# Константы для пагинации
ITEMS_PER_PAGE = 5

# Состояния FSM
class WarrantyStates(StatesGroup):
    enter_brand = State()
    enter_start_date = State()
    enter_duration = State()
    enter_duration_unit = State()
    edit_choice = State()
    edit_brand = State()
    edit_date = State()
    edit_duration = State()
    edit_duration_unit = State()

# Инициализация БД
def init_db():
    with sqlite3.connect('warranty.db') as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS warranties (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                brand TEXT NOT NULL,
                start_date TEXT NOT NULL,
                duration_days INTEGER NOT NULL,
                notified BOOLEAN DEFAULT FALSE
            )
        ''')

# Команда /start
@dp.message(Command("start"))
async def cmd_start(message: Message):
    await message.answer(
        "🔧 Бот для отслеживания сроков годности\n\n"
        "Доступные команды:\n"
        "/add - добавить наименование\n"
        "/list - ваш список наименований\n"
        "/edit - изменить наименование\n"
        "/delete - удалить наименование"
    )

# Добавление гарантии
@dp.message(Command("add"))
async def cmd_add(message: Message, state: FSMContext):
    await message.answer("Введите название товара или бренда:")
    await state.set_state(WarrantyStates.enter_brand)

@dp.message(WarrantyStates.enter_brand)
async def process_brand(message: Message, state: FSMContext):
    await state.update_data(brand=message.text)
    await message.answer("Введите дату начала использования (ГГГГ-ММ-ДД):")
    await state.set_state(WarrantyStates.enter_start_date)

@dp.message(WarrantyStates.enter_start_date)
async def process_date(message: Message, state: FSMContext):
    try:
        datetime.strptime(message.text, "%Y-%m-%d")
        await state.update_data(start_date=message.text)
        
        builder = InlineKeyboardBuilder()
        builder.button(text="Дни", callback_data="unit_days")
        builder.button(text="Недели", callback_data="unit_weeks")
        builder.button(text="Месяцы", callback_data="unit_months")
        builder.button(text="Годы", callback_data="unit_years")
        builder.adjust(2)
        
        await message.answer(
            "Выберите единицу измерения срока годности:",
            reply_markup=builder.as_markup()
        )
        await state.set_state(WarrantyStates.enter_duration_unit)
    except ValueError:
        await message.answer("❌ Неверный формат даты. Используйте ГГГГ-ММ-ДД")

@dp.callback_query(WarrantyStates.enter_duration_unit, F.data.startswith("unit_"))
async def process_duration_unit(callback: CallbackQuery, state: FSMContext):
    unit = callback.data.split("_")[1]
    await state.update_data(duration_unit=unit)
    await callback.message.edit_text(f"Введите срок годности в {unit}:")
    await state.set_state(WarrantyStates.enter_duration)
    await callback.answer()

@dp.message(WarrantyStates.enter_duration)
async def process_duration(message: Message, state: FSMContext):
    try:
        duration = float(message.text)
        if duration <= 0:
            raise ValueError
            
        data = await state.get_data()
        unit = data['duration_unit']
        
        # Конвертация в дни
        if unit == "days":
            duration_days = int(duration)
        elif unit == "weeks":
            duration_days = int(duration * 7)
        elif unit == "months":
            duration_days = int(duration * 30)  # Приблизительно
        elif unit == "years":
            duration_days = int(duration * 365)  # Приблизительно
        
        user_id = message.from_user.id
        
        with sqlite3.connect('warranty.db') as conn:
            conn.execute(
                "INSERT INTO warranties (user_id, brand, start_date, duration_days) VALUES (?, ?, ?, ?)",
                (user_id, data['brand'], data['start_date'], duration_days)
            )
        
        await message.answer("✅ Гарантия успешно добавлена!")
        await state.clear()
        await show_warranties(message, page=1)
        
    except ValueError:
        await message.answer("❌ Введите положительное число")

# --- Редактирование гарантии ---
@dp.message(Command("edit"))
async def cmd_edit(message: Message):
    await show_warranties(message, page=1, edit_mode=True)

@dp.callback_query(F.data.startswith("edit_"))
async def select_edit_field(callback: CallbackQuery, state: FSMContext):
    warranty_id = callback.data.split("_")[1]
    await state.update_data(warranty_id=warranty_id)
    
    builder = InlineKeyboardBuilder()
    builder.button(text="Название", callback_data="field_brand")
    builder.button(text="Дата начала", callback_data="field_date")
    builder.button(text="Срок годности", callback_data="field_duration")
    builder.button(text="❌ Отмена", callback_data="cancel_edit")
    builder.adjust(1)
    
    await callback.message.edit_text(
        f"Что вы хотите изменить для наименования?",
        reply_markup=builder.as_markup()
    )
    await state.set_state(WarrantyStates.edit_choice)
    await callback.answer()

@dp.callback_query(WarrantyStates.edit_choice, F.data.startswith("field_"))
async def select_field_to_edit(callback: CallbackQuery, state: FSMContext):
    field = callback.data.split("_")[1]
    await state.update_data(edit_field=field)
    
    if field == "brand":
        await callback.message.edit_text("Введите новое название:")
        await state.set_state(WarrantyStates.edit_brand)
    elif field == "date":
        await callback.message.edit_text("Введите новую дату начала (ГГГГ-ММ-ДД):")
        await state.set_state(WarrantyStates.edit_date)
    elif field == "duration":
        builder = InlineKeyboardBuilder()
        builder.button(text="Дни", callback_data="edit_unit_days")
        builder.button(text="Недели", callback_data="edit_unit_weeks")
        builder.button(text="Месяцы", callback_data="edit_unit_months")
        builder.button(text="Годы", callback_data="edit_unit_years")
        builder.adjust(2)
        await callback.message.edit_text(
            "Выберите единицу измерения срока годности:",
            reply_markup=builder.as_markup()
        )
        await state.set_state(WarrantyStates.edit_duration_unit)
    
    await callback.answer()

@dp.callback_query(WarrantyStates.edit_duration_unit, F.data.startswith("edit_unit_"))
async def process_edit_duration_unit(callback: CallbackQuery, state: FSMContext):
    unit = callback.data.split("_")[2]
    await state.update_data(duration_unit=unit)
    await callback.message.edit_text(f"Введите новый срок годности в {unit}:")
    await state.set_state(WarrantyStates.edit_duration)
    await callback.answer()

@dp.callback_query(WarrantyStates.edit_choice, F.data == "cancel_edit")
async def cancel_editing(callback: CallbackQuery, state: FSMContext):
    await callback.message.edit_text("❌ Редактирование отменено")
    await state.clear()
    await callback.answer()

@dp.message(WarrantyStates.edit_brand)
async def process_edit_brand(message: Message, state: FSMContext):
    data = await state.get_data()
    warranty_id = data['warranty_id']
    
    with sqlite3.connect('warranty.db') as conn:
        conn.execute(
            "UPDATE warranties SET brand = ? WHERE id = ?",
            (message.text, warranty_id)
        )
    
    await message.answer("✅ Название успешно обновлено!")
    await state.clear()
    await show_warranties(message, page=1)

@dp.message(WarrantyStates.edit_date)
async def process_edit_date(message: Message, state: FSMContext):
    try:
        datetime.strptime(message.text, "%Y-%m-%d")
        data = await state.get_data()
        warranty_id = data['warranty_id']
        
        with sqlite3.connect('warranty.db') as conn:
            conn.execute(
                "UPDATE warranties SET start_date = ?, notified = FALSE WHERE id = ?",
                (message.text, warranty_id)
            )
        
        await message.answer("✅ Дата начала успешно обновлена!")
        await state.clear()
        await show_warranties(message, page=1)
    except ValueError:
        await message.answer("❌ Неверный формат даты. Используйте ГГГГ-ММ-ДД")

@dp.message(WarrantyStates.edit_duration)
async def process_edit_duration(message: Message, state: FSMContext):
    try:
        duration = float(message.text)
        if duration <= 0:
            raise ValueError
            
        data = await state.get_data()
        warranty_id = data['warranty_id']
        unit = data['duration_unit']
        
        # Конвертация в дни
        if unit == "days":
            duration_days = int(duration)
        elif unit == "weeks":
            duration_days = int(duration * 7)
        elif unit == "months":
            duration_days = int(duration * 30)  # Приблизительно
        elif unit == "years":
            duration_days = int(duration * 365)  # Приблизительно
        
        with sqlite3.connect('warranty.db') as conn:
            conn.execute(
                "UPDATE warranties SET duration_days = ?, notified = FALSE WHERE id = ?",
                (duration_days, warranty_id)
            )
        
        await message.answer("✅ Срок годности успешно обновлен!")
        await state.clear()
        await show_warranties(message, page=1)
    except ValueError:
        await message.answer("❌ Введите положительное число")

# Список гарантий
@dp.message(Command("list"))
async def cmd_list(message: Message):
    await show_warranties(message, page=1)

async def show_warranties(message: Message | CallbackQuery, page: int = 1, edit_mode: bool = False):
    user_id = message.from_user.id if isinstance(message, Message) else message.message.chat.id
    today = datetime.now().date()
    
    with sqlite3.connect('warranty.db') as conn:
        cursor = conn.execute(
            "SELECT id, brand, start_date, duration_days FROM warranties WHERE user_id = ?",
            (user_id,)
        )
        warranties = cursor.fetchall()
    
    if not warranties:
        text = "ℹ️ У вас нет сохраненных наименований"
        if isinstance(message, Message):
            await message.answer(text)
        else:
            await message.message.edit_text(text)
        return
    
    # Пагинация
    total_items = len(warranties)
    total_pages = (total_items + ITEMS_PER_PAGE - 1) // ITEMS_PER_PAGE
    start_idx = (page - 1) * ITEMS_PER_PAGE
    end_idx = min(start_idx + ITEMS_PER_PAGE, total_items)
    
    text = f"📋 Ваш список (страница {page}/{total_pages}):\n\n"
    for warranty in warranties[start_idx:end_idx]:
        id_, brand, start_date, duration = warranty
        end_date = datetime.strptime(start_date, "%Y-%m-%d").date() + timedelta(days=duration)
        days_left = (end_date - today).days
        
        phrase = f"✅ Активно (осталось {days_left} дней)" if days_left >= 0 else f"❌ Истекло ({abs(days_left)} дней назад)"
        text += (
            f"Наименование: <b>{brand}</b>\n"
            f"Начало: {start_date}\n"
            f"Окончание: {end_date}\n"
            f"Статус: {phrase}\n\n"
        )
    
    builder = InlineKeyboardBuilder()
    
    if edit_mode:
        for warranty in warranties[start_idx:end_idx]:
            id_, brand, start_date, duration = warranty
            end_date = datetime.strptime(start_date, "%Y-%m-%d").date() + timedelta(days=duration)
            days_left = (end_date - today).days
            phrase = f"✅" if days_left >= 0 else f"❌"
            builder.button(text=f"✏️ {phrase} {brand} ({start_date} - {end_date})", callback_data=f"edit_{id_}")
    else:
        if page > 1:
            builder.button(text="⬅️ Назад", callback_data=f"page_{page-1}")
        if page < total_pages:
            builder.button(text="Вперед ➡️", callback_data=f"page_{page+1}")
    
    builder.adjust(1)
    
    if isinstance(message, Message):
        await message.answer(text, parse_mode="HTML", reply_markup=builder.as_markup())
    else:
        await message.message.edit_text(text, parse_mode="HTML", reply_markup=builder.as_markup())

@dp.callback_query(F.data.startswith("page_"))
async def process_page(callback: CallbackQuery):
    page = int(callback.data.split("_")[1])
    await show_warranties(callback, page=page)
    await callback.answer()

# Удаление гарантии
@dp.message(Command("delete"))
async def cmd_delete(message: Message):
    await show_warranties(message, page=1, delete_mode=True)

@dp.callback_query(F.data.startswith("delete_"))
async def process_delete(callback: CallbackQuery):
    warranty_id = callback.data.split("_")[1]
    
    with sqlite3.connect('warranty.db') as conn:
        conn.execute("DELETE FROM warranties WHERE id = ?", (warranty_id,))
    
    await callback.message.edit_text(f"✅ Наименование удалено!")
    await callback.answer()

# Проверка просроченных гарантий
async def check_expired_warranties():
    today = datetime.now().date()
    
    with sqlite3.connect('warranty.db') as conn:
        cursor = conn.execute(''
            SELECT id, user_id, brand FROM warranties 
            WHERE date(start_date, '+' || duration_days || ' days') < ? 
            AND notified = FALSE
        ''', (today.isoformat(),))
        
        expired = cursor.fetchall()
        
        for warranty in expired:
            id_, user_id, brand = warranty
            try:
                await bot.send_message(
                    chat_id=user_id,
                    text=f"⚠️ Срок годности на {brand} истек!"
                )
                conn.execute("UPDATE warranties SET notified = TRUE WHERE id = ?", (id_,))
            except Exception as e:
                logger.error(f"Ошибка уведомления: {e}")
        
        conn.commit()

# Фоновая задача
async def check_periodically():
    while True:
        await check_expired_warranties()
        await asyncio.sleep(86400)  # 24 часа

# Запуск бота
async def main():
    init_db()
    
    # Запускаем фоновую задачу
    asyncio.create_task(check_periodically())
    
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())

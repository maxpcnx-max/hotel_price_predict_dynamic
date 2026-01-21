import pandas as pd
import holidays
from datetime import timedelta
from utils import BASE_PRICES

def get_base_price(room_text):
    if not isinstance(room_text, str): return 0
    for key in BASE_PRICES:
        if key in room_text: return BASE_PRICES[key]
    return 0

def predict_segmented_price(model, start_date, n_nights, guests, r_code, res_code):
    MAX_CHUNK = 7 
    total_predicted = 0
    remaining_nights = n_nights
    current_date = start_date
    th_holidays = holidays.Thailand()
    
    while remaining_nights > 0:
        chunk_nights = min(remaining_nights, MAX_CHUNK)
        chunk_end_date = current_date + timedelta(days=chunk_nights)
        
        chunk_is_holiday = 0
        temp_date = current_date
        while temp_date < chunk_end_date:
            if temp_date in th_holidays:
                chunk_is_holiday = 1
                break
            temp_date += timedelta(days=1)
        
        chunk_is_weekend = 1 if current_date.weekday() in [5, 6] else 0
        
        inp_chunk = pd.DataFrame([{
            'Night': chunk_nights, 'total_guests': guests, 
            'is_holiday': chunk_is_holiday, 'is_weekend': chunk_is_weekend,
            'month': current_date.month, 'weekday': current_date.weekday(),
            'RoomType_encoded': r_code, 'Reservation_encoded': res_code
        }])
        
        chunk_price = model.predict(inp_chunk)[0]
        total_predicted += chunk_price
        remaining_nights -= chunk_nights
        current_date = chunk_end_date
    return total_predicted

def calculate_rule_based_price(base_per_night, start_date, n_nights, use_holiday, use_weekend):
    th_holidays = holidays.Thailand()
    total_price = 0
    current_date = start_date
    for _ in range(n_nights):
        multiplier = 1.0
        is_weekend = current_date.weekday() in [5, 6]
        is_holiday = current_date in th_holidays
        is_near_holiday = any((current_date + timedelta(days=i)) in th_holidays for i in range(1, 4))
        
        if is_holiday and use_holiday:
            multiplier = 1.7 if (is_weekend and use_weekend) else 1.5
        elif is_weekend and use_weekend:
            multiplier = 1.56 if (is_near_holiday and use_holiday) else 1.2
        elif is_near_holiday and use_holiday:
            multiplier = 1.3
        
        total_price += (base_per_night * multiplier)
        current_date += timedelta(days=1)
    return total_price

def calculate_clamped_price(model, start_date, n_nights, guests, r_code, res_code, room_name, use_h, use_w, hist_map):
    raw_predicted = predict_segmented_price(model, start_date, n_nights, guests, r_code, res_code)
    base_per_night = get_base_price(room_name)
    rule_price = calculate_rule_based_price(base_per_night, start_date, n_nights, use_h, use_w)
    
    hist_avg = hist_map.get(room_name, 0)
    if hist_avg > 0:
        offset = raw_predicted - (hist_avg * n_nights)
        final_price = rule_price + offset
    else:
        final_price = rule_price

    return max(final_price, base_per_night * n_nights), raw_predicted, rule_price
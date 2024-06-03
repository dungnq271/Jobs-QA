def modify_days_to_3digits(day=str):
    words = day.split()
    try:
        nday = int(words[0])
        return " ".join([f"{nday:03}"] + words[1:])
    except ValueError:
        return "9999"

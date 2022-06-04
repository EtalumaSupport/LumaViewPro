def wrong():
    try:
        'a'/2
        return True
    except:
        return False

print(wrong())

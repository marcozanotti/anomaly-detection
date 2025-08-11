
def is_weekend(dates):
    """Date is weekend"""
    return dates.dayofweek >= 5

def is_workday(dates):
    """Date is working day"""
    return dates.dayofweek < 5
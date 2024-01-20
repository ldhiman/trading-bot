class PortfolioManager:
    def __init__(self, initial_cash):
        self.cash = initial_cash
        self.positions = {}

    def buy(self, stock, quantity, price):
        cost = quantity * price
        if cost > self.cash:
            raise ValueError("Insufficient funds to buy")
        self.cash -= cost
        if stock in self.positions:
            self.positions[stock] += quantity
        else:
            self.positions[stock] = quantity

    def sell(self, stock, quantity, price):
        if stock not in self.positions or self.positions[stock] < quantity:
            raise ValueError("Insufficient stock to sell")
        revenue = quantity * price
        self.cash += revenue
        self.positions[stock] -= quantity
        if self.positions[stock] == 0:
            del self.positions[stock]

    def get_portfolio_value(self, stock_prices):
        total_value = self.cash
        for stock, quantity in self.positions.items():
            if stock in stock_prices:
                total_value += quantity * stock_prices[stock]
        return total_value

    def print_portfolio_status(self):
        print(f"Portfolio Status: Cash = {self.cash}")
        for stock, quantity in self.positions.items():
            print(f"{stock} Units = {quantity}")

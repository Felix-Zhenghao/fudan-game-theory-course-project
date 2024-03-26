############################################################################################################
# TO DO: 1. Test with massive data.
#        2. Calculate by hand whether the results are correct.
#        3. Improve the robustness of the user_input function.
#        4. Implement the 'learning_player' code to study the effect of multi-agent learning in the game.
############################################################################################################


from __future__ import annotations

import numpy as np
# import matplotlib.pyplot as plt

# Helper function(s)

def array_to_dict(name_list: list,
                  value_array: np.ndarray):
    """
    This function turn the ndarray into a dict.
    The key is given by the input name_list, representing the name of all players.
    It is used to print more readable game result.

    Example: 
    array_to_dict( ["player_1", "player_2", "player_3"] , np.array([1,2,3]) )
    >> {"player_1":1, "player_2":2, "player_3":3}
    """

    assert len(name_list)==len(value_array), "Wrong array_to_dict call: The lenght of name_list does not match the length of array"

    return {name: value for name, value in zip(name_list, value_array)}

def get_element(target_list: list,
                index_list: list):
    
    return np.array([target_list[i] for i in index_list])

def print_save(text, filename="game_log.txt"):
    """
    Prints the given text to the console and saves it to a specified file.
    
    :param text: The text to print and save. This can be a formatted string.
    :param filename: The name of the file where the text will be saved. Defaults to 'output.txt'.
    """
    # Print the text to the console
    print(text)
    
    # Open the file in append mode and write the text
    with open(filename, "a") as file:
        file.write(text + "\n")

    
###################################################
# The class of definition of one round of the game.
###################################################

class OneRoundGame:
    # TO DO: enable collaboration.

    def __init__(self, 
                 num_player: int = 3, 
                 collaboration: bool = False, 
                 demand_shock: bool = False,
                 player_name: list = None
                 ):
        
        self.num_player = num_player
        self.collaboration = collaboration
        self.demand_shock = demand_shock

        # Default player names: ["player_1","player_2",...]
        if player_name is None:
            self.player_name = ['player_{}'.format(i) for i in range(1, num_player + 1)]
        else: self.player_name = player_name
        assert len(self.player_name)==self.num_player, "Please provide enough player names or don't provide names to use default names."


    def calculateMarketShare(self, price: np.ndarray):
        """Calculate market share according to the price offer. The result'll be passed to calculateDemand method."""

        sum_price = sum(price)
        t = np.array([np.exp(6-18*i/sum_price) for i in price])
        return t/np.sum(t)

    
    def calculateTotalDemand(self, 
                             price: np.ndarray, 
                             enable_shock: bool = False
                             ):
        """
        Calculate total demand according to the price offer.
        The arg 'enable_shock' is to ensure that the shock isn't enforced twice if we re-calculate the demand after demand unsatisfaction.
        When this function is called after demand unsatisfaction, enable_shock is set as False (the default). 
        First calculation of demand should set it as True.
        """

        # assert len(price)==self.num_player , "The number of price offer does not match number of players"

        market_share = self.calculateMarketShare(price)
        weighted_price = np.vdot(market_share,price)

        total_demand = 2000-20*weighted_price # It will return datatype numpy.int64 in default
        if self.demand_shock & enable_shock:
            total_demand += np.random.uniform(low = 0.0,high=100.0) # The demand shock is sampled from U(0,100)

        return total_demand


    def allocateDemand(self, total_demand, price):
        """
        Input: the total_demand calculated by self.calculateTotalDemand and the price offer array.
        Output: the allocated demand array.
        To ensure the total_demand > 0, there need to be an outer condition to control it.
        """

        market_share = self.calculateMarketShare(price)
        demand = total_demand * market_share

        return np.rint(demand) # Round the demand to the nearest integer.
    

    def calculateSales(self, 
                       production: np.ndarray, 
                       price: np.ndarray, 
                       inventory: np.ndarray
                       ):

        total_demand = self.calculateTotalDemand(price=price,enable_shock=self.demand_shock)
        demand = self.allocateDemand(total_demand=total_demand,price=price)
        max_sales = production + inventory
        demand_unsatisfaction = demand >= max_sales # If demand is not satisfied <and> excess demand haven't been allocated, set as False.

        sales = np.minimum(max_sales,demand)

        check = np.ones(self.num_player) # All ones array
        chack_ = np.sum(check * demand_unsatisfaction) != self.num_player
        while (np.sum(check * demand_unsatisfaction) != self.num_player) & (np.sum(check * demand_unsatisfaction) != 0):
        # If some demand is satisfied and some is not, do:
            
            print_save(f"total_demand:{total_demand}\n")

            index = np.arange(self.num_player) 
            index = index[demand_unsatisfaction == False].tolist() # Find all players with False demand_unsatisfaction

            tmp_price = get_element(target_list=price,index_list=index)
            tmp_production = get_element(target_list=production,index_list=index)
            tmp_inventory = get_element(target_list=inventory,index_list=index)
            tmp_sales = get_element(target_list=sales,index_list=index)

            # One round of unsatisfied need allocation
            total_demand = np.maximum(self.calculateTotalDemand(price=tmp_price) - np.sum(sales), 0.0)
            if np.rint(total_demand) == 0:
                return sales # The market has no need given the left producers in the market.
            demand =  self.allocateDemand(total_demand=total_demand,price=tmp_price)

            tmp_max_sales = tmp_production + tmp_inventory - tmp_sales # Find the amount of allowed maximum sales.

            for i,j in zip(index,range(len(index))):
                demand_unsatisfaction[i] = demand[j] >= tmp_max_sales[j] # Update demand states.
                sales[i] += np.minimum(tmp_max_sales[j],demand[j]) # Update sales.

        print_save(f"#####Your sales performance of this period#####\n{array_to_dict(self.player_name,sales)}\n")
        print_save(f"#####The market share of each player in this period#####\n{array_to_dict(self.player_name,sales/np.sum(sales))}\n")
        return sales
    
    
    def updateInventory(self, 
                        inventory: np.ndarray, 
                        production: np.ndarray, 
                        sales: np.ndarray
                        ):
        print_save(f"#####Updated inventory after this period#####\n{array_to_dict(self.player_name,inventory)}\n")
        return np.maximum(inventory+production-sales,0.0)
    
    
    def calculateCost(self,
                      cumulative_production: int = None,
                      learning: bool = True,
                      learning_rate: float = 0.86,
                      production: np.ndarray = None,
                      ) -> np.ndarray:
        # TO DO: try to make the cost func more flexible by adding func: callable=... in the arg.

        if learning:
            assert learning_rate is not None, "please provide learning_rate \in (0,1] to enable learning"
            
            def alpha(learning_rate, accumlate_prodcution):
                return np.power(learning_rate,np.log2(accumlate_prodcution))
            
            alpha = alpha(learning_rate=learning_rate, accumlate_prodcution=cumulative_production)
            marginal_cost = 100 * alpha + 0.1 * production
            total_cost = 5000 + 100 * alpha * production + 0.05 * np.power(production,2)

        else:
            marginal_cost = 0.1 * production
            total_cost = 5000 + 100 * production + 0.05 * np.power(production,2)

        average_cost = total_cost/production
        cumulative_production += production

        print_save(f"#####The total cost (w/ inventory) for all players this period#####\n{array_to_dict(self.player_name,total_cost)}\n")
        print_save(f"#####The average cost for all players this period#####\n{array_to_dict(self.player_name,average_cost)}\n")
        print_save(f"#####The marginal cost for all players#####\n{array_to_dict(self.player_name,marginal_cost)}\n")
        print_save(f"#####The updated cumulative production level for all players after this period#####\n{array_to_dict(self.player_name,cumulative_production)}\n")

        cost = {"TC":total_cost,"AC":average_cost,"MC":marginal_cost}

        return cost, cumulative_production
    
    def calculateProfit(self,
                        cost: dict,
                        inventory: np.ndarray,
                        sales: np.ndarray,
                        price: np.ndarray,
                        cumulative_profit: np.ndarray
                        ):
        
        # Carry cost of current period's inventory is immedietely incurred. Thus, the input inventory should have been updated.
        inventory_cost = 2 * inventory
        total_cost = cost["TC"] + inventory_cost
        print_save(f"#####The inventory cost for all players this period#####\n{array_to_dict(self.player_name,inventory_cost)}\n")
        print_save(f"#####The total cost (w inventory) for all players this period#####\n{array_to_dict(self.player_name,total_cost)}\n")

        total_profit = sales * price - total_cost
        cumulative_profit += total_profit
        print_save(f"#####The total profit for all players this period#####\n{array_to_dict(self.player_name,total_profit)}\n")
        print_save(f"#####The cumulative profit for all players after period#####\n{array_to_dict(self.player_name,cumulative_profit)}\n")
        profit = {"profit_this_period":total_profit,"cumulative_profit_after_this_period": cumulative_profit}
        
        return profit
    

    def playRound(self,
                  previous_inventory: np.ndarray,
                  cumulative_production: np.ndarray,
                  cumulative_profit: np.ndarray,
                  production: np.ndarray,
                  price: np.ndarray,
                  game_index: int
                  ):
        """
        The interface to conveniently play one round of the game and get the round result.

        ==== 1. Input ====

        *** 1.1 Previous states ***
        - previous_inventory: inventory at the end of the last period 
        - cumulative_production: cumulative production of all previous periods
        - cumulative_profit: cumulative profit of all previous periods

        *** 1.2 Current player offer ***
        - production: production level of all players in this period
        - price: price offer of all players in this period

        *** 1.3 Game configuration ***
        - game_index: the round index of this round of game

        ==== 2. Output ====
        - inventory: inventory level after this period of game
        - production: cumulative procution after this period of game
        - profit: cumulative profit after this period of game
        """

        # Print the game configuration for the sake of readability.
        print_save(f"\n\n#########################################\n## ROUND {game_index} of the game starts\n#########################################\n")
        print_save(f"===============================\nThere are  {self.num_player}  players in this round.\nTheir names: {self.player_name}\nTheir production levels: {array_to_dict(self.player_name,production)}\nTheir price offers:  {array_to_dict(self.player_name,price)}\n===============================\n\n")


        # Calculate and allocate the market demand according to the price offer. (Implemented in the self.calculateSales)
        # Calculate the sales of all players according to the market demand, inventory and price offer.
        sales = self.calculateSales(production=production,price=price,inventory=previous_inventory)

        # Calculate costs and profits. Update cumulative states like inventory, production, profit.
        inventory = self.updateInventory(inventory=previous_inventory,production=production,sales=sales)
        costs, production = self.calculateCost(cumulative_production=cumulative_production,production=production)
        profit = self.calculateProfit(cost=costs,inventory=inventory,sales=sales,price=price,cumulative_profit=cumulative_profit)["cumulative_profit_after_this_period"]

        # Return inventory, production, profit to later rounds of the game.
        return inventory, production, profit



class playGame:
    # TO DO: enable playing different class.
    """Here we play several rounds of OneRoundGame."""
    
    def __init__(self,
                 initial_cumulative_production: np.ndarray,
                 initial_inventory: np.ndarray = None,
                 round: int = 20,
                 num_player: int = 3,
                 collaboration: bool = False,
                 demand_shock: bool = False,
                 player_name: list = None
                 ):
        
        self.round = round
        self.num_player = num_player
        self.initial_cumulative_production = initial_cumulative_production
        self.collaboration = collaboration
        self.demand_shock = demand_shock
        
        if initial_inventory is not None:
            self.initial_inventory = initial_inventory
        else: self.initial_inventory = np.zeros(self.num_player,dtype=float)

        # Default player names: ["player_1","player_2",...]
        if player_name is None:
            self.player_name = ['player_{}'.format(i) for i in range(1, num_player + 1)]
        else: self.player_name = player_name
        assert len(self.player_name)==self.num_player, "Please provide enough player names or don't provide names to use default names."

        self.initial_game = OneRoundGame(num_player=self.num_player,
                                         collaboration=self.collaboration,
                                         demand_shock=self.demand_shock,
                                         player_name=self.player_name)

    def user_input(self):
        # Initialize an empty list to store the price offers
        price_offers = []
        production_levels = []

        # Ask for user input for each player's price offer and production level
        for i in self.player_name:

            # Prompt the user for input
            confirmation = False
            while not confirmation: 
                offer = float(input(f"Input the price offered by {i}: "))
                prod = float(input(f"Input the production level of {i}: "))
                confirmation = input(f"The price offer and production level of {i} is {offer} and {prod}, respectively. If you confirm the value, print y. If you want to rewrite the input, type anything and press 'Enter':")
                confirmation = confirmation=='y'

            # Append the offer to the list
            price_offers.append(offer)
            production_levels.append(prod)

        return np.array(price_offers), np.array(production_levels)
    
    def play(self):
        # TO DO: Write the printed result into a txt file.

        price_input, production_input = self.user_input()
        
        inventory, production, profit = self.initial_game.playRound(previous_inventory=self.initial_inventory,
                                                                    cumulative_production=self.initial_cumulative_production,
                                                                    cumulative_profit=np.zeros(self.num_player,dtype=float),
                                                                    production=production_input,
                                                                    price=price_input,
                                                                    game_index=1)

        for i in range(self.round - 1):

            game = OneRoundGame(num_player=self.num_player,demand_shock=self.demand_shock,player_name=self.player_name) # Create the round.

            price_input, production_input = self.user_input() # Give price and production of each player in this round

            inventory, production, profit = game.playRound(previous_inventory=inventory,
                                                           cumulative_production=production,
                                                           cumulative_profit=profit,
                                                           production=production_input,
                                                           price=price_input,
                                                           game_index=i+2) # Play!!



game = playGame(initial_cumulative_production=np.array([32.,32.,32.]),
                initial_inventory=np.zeros(3,dtype=float),
                round = 3,
                num_player=3,
                collaboration=False,
                demand_shock=True,
                player_name=["Adam","Felix","Q神"])


game.play()




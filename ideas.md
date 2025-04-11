1. Idea for this fetching file:
    sync_deribit_live.py	Runs every 15m. Triggers incremental updates to all above.
    IDEA: first get all the information that is missing by using the api and then when it is up to date, switch to websocket to make sure the data is always immediately fetched.


Next Steps

1. Hyperparameter Optimization: Consider running a hyperparameter sweep using W&B to further improve model performance
2. Portfolio Optimization: Enhance the model with cross-instrument portfolio awareness
3. Online Learning: Implement continuous training with experience replay for adaptation to new market conditions
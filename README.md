# Proton Data Science Challenge v1

Congratulations, you've passed our technical interview! Now it's time for a more in-depth chance to get a sense of the problems we solve and how we solve them.

We've built a small model here for predicting items people should add to their shopping basksets. We'll give you instructions separately on how we'd like you to improve it.


# FAQ

## 1. What does the function `internal_utils.get_product_count(client_name)` return?
This function will return an integer representing the total number of products that a specific client(distributor) carries in their product catalog.\
For example, say Bob's Distribution Company works with restaurants and carries three products:
Olive Oil, Tomato Sauce and Pizza dough.\
In this case `internal_utils.get_product_count("bobs_distribution_company")` will return the integer 3.\
For the purposes of the challenge you can assume the number will always be between 100k - 500k and that this number will be different for each client

## 2. Client and customer. What means what?
Great question! This can sometimes be a source of confusion especially when those terms show up together very often.\
When we say **client** we mean one of our clients, i.e. a distributor. When we say **customer** we refer to one of our 
client's (distributor's) customers. 

import sim, model

from statistics import mean, median

initial_data, scores = sim.play(iterations=15000, render=False, dumb=True)

MIN_SCORE = 50
revised_scores = [x for x in scores if x > MIN_SCORE]
revised_data = [y for i,x in enumerate(initial_data) if scores[i] > MIN_SCORE for y in x]

print("Mean: {}, Median: {}".format(mean(scores), median(scores)))
print("Rev Mean: {}, Rev Median: {}".format(mean(revised_scores), median(revised_scores)))

model_v0 = model.train(revised_data)

_, trained_scores = sim.play(iterations=5, render=True, model=model_v0, iter_sleep = 1.0)

print("Test Mean: {}, Test Median: {}".format(mean(trained_scores), median(trained_scores)))


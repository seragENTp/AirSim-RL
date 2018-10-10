import tensorflow as tf 

import numpy as np

from environment import Environment



from model import DQNetwork

env = Environment()


DQNetwork = DQNetwork(learning_rate = 0)


with tf.Session() as sess :

	total_test_rewards = []

	saver = tf.train.Saver()


	saver.restore(sess, "./models/model.ckpt")

	for episode in range(1):

		total_rewards = 0


		state = env.reset()

		done = False

		while not done:


			state = state.reshape((1,state.shape[0],state.shape[1],state.shape[2]))

			Qs = sess.run (DQNetwork.output,feed_dict = {DQNetwork.inputs_: state})

			action = np.argmax(Qs)

			new_state,reward,done = env.step(action)

			total_rewards+= reward

			state = new_state



		print ("Episode %s",episode, "total_rewards %s",total_rewards)
		






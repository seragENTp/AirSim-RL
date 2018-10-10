import numpy as np

import tensorflow as tf

import math

import random

import numpy as np

import matplotlib

import matplotlib.pyplot as plt

import cv2

from model import ReplayMemory,DQNetwork

from environment import Environment

import sys


#from utilities import 


class AgentTrainer():
	def __init__(self):

		self.BATCH_SIZE = 128
		self.GAMMA = 0.99
		self.EPS_START = 1.0
		self.EPS_END = 0.05
		self.EPS_DECAY = 0.000005
		self.TARGET_UPDATE = 5

		self.pretrain_length = self.BATCH_SIZE
		#self.state_size = [55,3]
		
		self.action_size = 3
		self.hot_actions = np.array(np.identity(self.action_size).tolist())
		#self.action_size = len(self.hot_actions)
		self.learning_rate = 0.0005
		#self.total_episodes = 12
		self.max_steps = 1000

		self.env = Environment()

		self.memory_maxsize = 10000

		self.DQNetwork = DQNetwork(learning_rate = self.learning_rate,name = 'DQNetwork')

		self.TargetNetwork = DQNetwork(learning_rate = self.learning_rate , name = 'TargetNetwork')

		self.memory = ReplayMemory(max_size=self.memory_maxsize)

		self.saver = tf.train.Saver()

		#self.TargetUpdate = update_target_graph()

	def select_action(self, sess,decay_step, state, actions):
		## EPSILON GREEDY STRATEGY Choose action a from state s using epsilon greedy.
		## First we randomize a number
		exp_exp_tradeoff = np.random.rand()

		
		explore_probability = self.EPS_END + (self.EPS_START - self.EPS_END) * np.exp(-self.EPS_DECAY * decay_step)

		if (explore_probability > exp_exp_tradeoff):
			# Make a random action (exploration)
			choice = random.randint(1,len(self.hot_actions))-1
			action = self.hot_actions[choice]
			#print('action_taken is random',action)

		else:
			# Get action from Q-network (exploitation)
			# Estimate the Qs values state
								
			Qs = sess.run(self.DQNetwork.output, feed_dict = {self.DQNetwork.inputs_: state.reshape((1,) + state.shape)})


			# Take the biggest Q value (= the best action)
			choice = np.argmax(Qs)
			action = self.hot_actions[choice]
			
	    
	    
		return action, explore_probability

	#This function helps us to copy one set of variables to another
	
	
	def update_target_graph(self):

		
		return op_holder

	def train(self,num_episodes,sess):

		# Instantiate memory
		#memory = Memory(max_size = memory_size)
		for i in range(self.pretrain_length):
			# If it's the first step
			if i == 0:
				state = self.env.reset()
			    
			    		    
			# Get the next_state, the rewards, done by taking a random action
			choice = random.randint(1,len(self.hot_actions))-1
			action = self.hot_actions[choice]
			next_state, reward, done= self.env.step(np.argmax(action))
			

			# If the episode is finished (we're dead 3x)
			if done:
				# We finished the episode
				next_state = np.zeros(state.shape)
				# Add experience to memory
				self.memory.add((state, action, reward, next_state, done))
				# Start a new episode
				state = self.env.reset()
			    
			    
			    
			else:
			    # Add experience to memory
				self.memory.add((state, action, reward, next_state, done))
				#print("adding to memory")
				sys.stdout.flush()
				# Our new state is now the next_state
				state = next_state
		


		decay_step = 0

		rewards_list = []

		total_steps = 0

		
		for episode in range(num_episodes):



			#print('epidose',episode)


			# Set step to 0
			step = 0

			total_reward = 0

			# Initialize the rewards of the episode
			episode_rewards = []

			# Make a new episode and observe the first state
			state = self.env.reset()

			done = False

			#cv2.imshow(state)
			#cv2.waitKey(100)

			
			while not done:
				step += 1

				total_steps+=1
				
				
				
				#Increase decay_step
				decay_step +=1

				# Predict the action to take and take it

				action, explore_probability = self.select_action(sess,decay_step, state, self.hot_actions)

				#Perform the action and get the next_state, reward, and done information

				
				next_state, reward, done = self.env.step(np.argmax(action))


				

				# Add the reward to total reward
				episode_rewards.append(reward)

				# If the game is finished
				if done:
				    # The episode ends so no next state
					next_state = np.zeros(state.shape, dtype=np.int)

					
					steps_taken = step
					
					# Get the total reward of the episode
					total_reward = np.sum(episode_rewards)

					

					rewards_list.append((episode, total_reward))

					# Store transition <st,at,rt+1,st+1> in memory D
					self.memory.add((state, action, reward, next_state, done))

				else:
					# Stack the frame of the next_state
					# next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)

					# Add experience to memory
					self.memory.add((state, action, reward, next_state, done))

					steps_taken = step

					

					# st+1 is now our current state
					state = next_state
	                

		            ### LEARNING PART            
	            # Obtain random mini-batch from memory
				batch = self.memory.sample(self.BATCH_SIZE)
				states_mb = np.array([each[0] for each in batch], ndmin=3)
				actions_mb = np.array([each[1] for each in batch])
				rewards_mb = np.array([each[2] for each in batch]) 
				next_states_mb = np.array([each[3] for each in batch], ndmin=3)
				dones_mb = np.array([each[4] for each in batch])

				target_Qs_batch = []

								
				# Get Q values for next_state 
				Qs_next_state = sess.run(self.DQNetwork.output, feed_dict = {self.DQNetwork.inputs_: next_states_mb})
				# Calculate Qtarget for all actions that state
				q_target_next_state = sess.run(self.TargetNetwork.output, feed_dict = {self.TargetNetwork.inputs_: next_states_mb})

				# Set Q_target = r if the episode ends at s+1, otherwise set Q_target = r + gamma*maxQ(s', a')
				for i in range(0, len(batch)):
					terminal = dones_mb[i]

					# If we are in a terminal state, only equals reward
					if terminal:
						target_Qs_batch.append(rewards_mb[i])

					else:
						target = rewards_mb[i] + self.GAMMA * np.max(q_target_next_state[i])
						#print (target)
						#print(Qs_next_state[i])
						target_Qs_batch.append(target)

				targets_mb = np.array([each for each in target_Qs_batch])

				#print(targets_mb)

				loss, _ = sess.run([self.DQNetwork.loss, self.DQNetwork.optimizer],
				                        feed_dict={self.DQNetwork.inputs_: states_mb,
				                                   self.DQNetwork.target_Q: targets_mb,
				                                   self.DQNetwork.actions_: actions_mb})


			if episode%self.TARGET_UPDATE==0:
				# Update the parameters of our TargetNetwork with DQN_weights
				#update_target = self.TargetUpdate()


				# Get the parameters of our DQNNetwork
				from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "DQNetwork")

				# Get the parameters of our Target_network
				to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "TargetNetwork")

				op_holder = []

				# Update our target_network parameters with DQNNetwork parameters
				for from_var,to_var in zip(from_vars,to_vars):
					op_holder.append(to_var.assign(from_var))
				
				sess.run(op_holder)

				print("Target Model updated")

	            # Write TF Summaries
	           # summary = sess.run(write_op, feed_dict={self.DQNetwork.inputs_: states_mb,
	            #                                       self.DQNetwork.target_Q: targets_mb,
	            #                                       self.DQNetwork.actions_: actions_mb})
	           # writer.add_summary(summary, episode)
	           # writer.flush()

			print('Total steps: {}'.format(total_steps),'Episode: {}'.format(episode),'Step: {}'.format(steps_taken),
					              'Total reward: {}'.format(np.sum(episode_rewards)),
					              'Explore P: {:.4f}'.format(explore_probability),
					            'Training Loss {:.4f}'.format(loss))


	 		# Save model every 100 episodes
			if episode % 100 == 0:
				save_path = self.saver.save(sess, "./models/model.ckpt")
				print("Model Saved")






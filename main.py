import tensorflow as tf

from trainer import AgentTrainer



Trainer = AgentTrainer()

restore_session = False

saver = tf.train.Saver()

with tf.device('/gpu:1'):

	with tf.Session() as sess :


		sess.run(tf.global_variables_initializer())

		Trainer.train(100000,sess)
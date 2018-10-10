import tensorflow as tf

from trainer import AgentTrainer



Trainer = AgentTrainer()

restore_session = False

saver = tf.train.Saver()

with tf.device('/gpu:1'):

	with tf.Session() as sess :

		if restore_session :

			
			saver.restore(sess, "./models/model.ckpt")
			print("Model restored.")
			Trainer.train(1000,sess)


		else :

			sess.run(tf.global_variables_initializer())

			Trainer.train(100000,sess)
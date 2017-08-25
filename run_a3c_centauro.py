import multiprocessing
import threading
import tensorflow as tf
from environment import centauro_env
from a3c import a3c_agent, a3c_net

N_WORKERS = 2 # multiprocessing.cpu_count()

if __name__ == "__main__":
    SESS = tf.Session()

    with tf.device("/gpu:0"):
        GLOBAL_AC = a3c_net.ACNet(SESS, a3c_net.GLOBAL_NET_SCOPE)  # we only need its params
        workers = []
        summary_writer = tf.summary.FileWriter('data/log', SESS.graph)
        # Create worker
        for i in range(N_WORKERS):
            i_name = 'W_%i' % i   # worker name
            env = centauro_env.Simu_env(20000 + i)
            workers.append(a3c_agent.Worker(SESS, i_name, env, summary_writer, GLOBAL_AC))

    COORD = tf.train.Coordinator()
    SESS.run(tf.global_variables_initializer())

    saver = tf.train.import_meta_graph('./model/grid_feature_model.cptk.meta')
    saver.restore(SESS,tf.train.latest_checkpoint('./model'))
    graph = tf.get_default_graph()

    auto_params_trained = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Global_Net/auto')
    pull_auto_params_op = [l_p.assign(g_p) for l_p, g_p in zip(GLOBAL_AC.auto_params, auto_params_trained)]
    SESS.run(pull_auto_params_op)

    summary_writer = tf.summary.FileWriter('data/log', SESS.graph)
    saver = tf.train.Saver()

    # print ('Loading Model...')
    # ckpt = tf.train.get_checkpoint_state('./data/')
    # if ckpt and ckpt.model_checkpoint_path:
    #     saver.restore(SESS, ckpt.model_checkpoint_path)
    #     print ('loaded')
    # else:
    #     print ('no model file')    

    saver.save(SESS, './data/model.cptk') 

    worker_threads = []
    for worker in workers:
        job = lambda: worker.work(saver)
        t = threading.Thread(target=job)
        t.start()
        worker_threads.append(t)
    COORD.join(worker_threads)
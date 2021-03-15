#!/usr/bin/env python3

from keras import callbacks as cbks
import numpy as np

def fit_models(callback_model,
               models,
               generators,
               metrics_names,
               batch_size,
               validation_img_gen=None,
               steps_per_epoch=None,
               epochs=1,
               verbose=1,
               callbacks=None,
               initial_epoch=0):
    epoch = initial_epoch

    # Prepare display labels.
    callback_metrics = [n for m in metrics_names for n in m.keys()]

    # prepare callbacks
    stateful_metric_names = []
    for model in models:
        model.history = cbks.History()
#        try:
#            stateful_metric_names.extend(model.stateful_metric_names)
#        except AttributeError:
#            stateful_metric_names.extend(model.model.stateful_metric_names)
    _callbacks = [cbks.BaseLogger(
        stateful_metrics=stateful_metric_names)]
    if verbose:
        _callbacks.append(
            cbks.ProgbarLogger(
                count_mode='steps',
                stateful_metrics=stateful_metric_names))
    _callbacks += (callbacks or []) + [model.history for model in models]
    callbacks = cbks.CallbackList(_callbacks)

    # it's possible to callback a different model than self:
    callbacks.set_model(callback_model)
    callbacks.set_params({
        'epochs': epochs,
        'steps': steps_per_epoch,
        'verbose': verbose,
        'do_validation': False,
        'metrics': callback_metrics,
    })
    callbacks.on_train_begin()

    try:
        callback_model.stop_training = False
        # Construct epoch logs.
        epoch_logs = {}
        while epoch < epochs:
#            for model in models:
#                try:
#                    stateful_metric_functions = model.stateful_metric_functions
#                except AttributeError:
#                    stateful_metric_functions = model.model.stateful_metric_functions
#                for m in stateful_metric_functions:
#                    m.reset_states()
            if validation_img_gen:
                X_rgb_batch, X_depth_batch, y_label = next(validation_img_gen)
                callbacks.on_predict_begin([X_rgb_batch, X_depth_batch])
            callbacks.on_epoch_begin(epoch)
            steps_done = 0
            batch_index = 0
            while steps_done < int(steps_per_epoch/4):
                


                # build batch logs
                batch_logs = {}
                batch_logs['batch'] = batch_index
                batch_logs['size'] = batch_size
                callbacks.on_batch_begin(batch_index, batch_logs)

                for model, output_generator, metrics in zip(models, generators, metrics_names):

                    generator_output = next(output_generator)

                    if not hasattr(generator_output, '__len__'):
                        raise ValueError('Output of generator should be '
                                         'a tuple `(x, y, sample_weight)` '
                                         'or `(x, y)`. Found: ' +
                                         str(generator_output))
#                    print('\n')
#                    print(len(generator_output))
                    if len(generator_output) == 2:
                        x, y = generator_output
                        
                        sample_weight = None
                    elif len(generator_output) == 3:
                        x, y, sample_weight = generator_output

                    else:
                        raise ValueError('Output of generator should be '
                                         'a tuple `(x, y, sample_weight)` '
                                         'or `(x, y)`. Found: ' +
                                         str(generator_output))

                    outs = model.train_on_batch(x, y, sample_weight=sample_weight)

                    if not isinstance(outs, list):
                        outs = [outs]

                    for name, i in metrics.items():
                        batch_logs[name] = outs[i]
                    if model.name == 'generator_train':
#                        print('\n callback for print')
#                        print('\n length of x:{}'.format(len(x)))
#                        X_rgb_batch, X_depth_batch = x
                        batch = [x[0],y[-1]]
                        callbacks.on_batch_end(batch = batch, logs=batch_logs)     
#                        print('batch print')
                    else:
#                        print('\n callback w/o print')
                        callbacks.on_batch_end([],logs= batch_logs)

                batch_index += 1
                steps_done += 1
                
                # Epoch finished.
                
                if callback_model.stop_training:
                    break
            print('Steps done:{}'.format(steps_done))
            print('epoch finished')
            if validation_img_gen:
                X_rgb_batch, X_depth_batch, y_label = next(validation_img_gen)
                callbacks.on_predict_begin([X_rgb_batch, X_depth_batch])

            callbacks.on_epoch_end(epoch, epoch_logs)
            epoch += 1
            
            if callback_model.stop_training:
                break

    finally:
        pass

    callbacks.on_train_end()

    return [model.history for model in models]

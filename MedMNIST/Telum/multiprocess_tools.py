from multiprocessing import Process, Queue, cpu_count

q = Queue()

def chunks(l, n):
    cl = len(l)//n
    cr = len(l)%n
    if cr > 0:
        cl +=1
    return _chunks(l,cl)

def _chunks(l, n):
    for i in range(0,len(l), n):
        yield l[i:i+n]

def inference(file_list, model, classes, rank, loader_procedure, inf_procedure, batch_size):
    file_list_c = list(_chunks(file_list, batch_size))
    results = []
    labels = []
    for batch in range(len(file_list_c)):
        image_data, labels_ = loader_procedure(file_list_c[batch], classes)
        results_ = inf_procedure(image_data, model, classes)

    for a in range(len(results_)):
        results.append(results_[a])
        labels.append(labels_[a])

    q.put({"results":results, "labels":labels})

def mp_inference(file_list, classes, model, loader_procedure, inf_procedure, nproc, batch_size):
    chunked_file_list = list(chunks(file_list, nproc))
    processes = []

    for a in range(nproc):
        processes.append(Process(target=inference, args=(chunked_file_list[a], model, classes, a, loader_procedure, inf_procedure, batch_size)))
        processes[a].start()	

    outputs = []
    labels = []
    for a in range(nproc):
        message = q.get()
        for b in message["results"]:
            outputs.append(b)
        for c in message["labels"]:
            labels.append(c)

    for a in range(nproc):
        print(f"Joining {a}")
        processes[a].join()

    print(f"Done")

    return outputs, labels

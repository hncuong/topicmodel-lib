import os, sys, logging
from Tkinter import *
import functools
import numpy as np

sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/' + '../..'))

from tmlib.lda import LdaModel
from tmlib.datasets import utilizies
from tmlib.datasets.utilizies import DataFormat
from database import DataBase

def calc_similar(vector_i, vector_j, thresh):
    indexs = np.where(vector_i <= thresh)[0]
    vector_i[indexs] = 0
    result = list()
    for vector in vector_j:
        indexs = np.where(vector <= thresh)[0]
        vector[indexs] = 0
        indexs = np.where(vector*vector_i > thresh)[0]
        diff = np.fabs(np.log(vector_i[indexs]) - np.log(vector[indexs]))
        result.append(np.sum(diff))
    return np.array(result)

def frac(n):
    if n < 0:
        n += 2*np.pi
    return 360 * n / (2*np.pi)

def is_in_arc(x, y, centerx, centery, R, angle0, angle1):
    if (x-centerx)**2 + (y-centery)**2 > R**2:
        return False
    theta = - np.arctan2(y-centery, x-centerx)
    return angle0 <= frac(theta) <= angle1


def get_arc(x, y, origin_coord, R, list_angle):
    for i in range(len(list_angle)):
        angle0, angle1 = list_angle[i]
        if is_in_arc(x, y, origin_coord[0], origin_coord[1], R, angle0, angle1):
            return i

    return None


class TopicPage(Frame):

    def __init__(self, root, model, db, vocab_file):
        Frame.__init__(self, root)

        self.top_words = model.print_top_words(20, vocab_file)
        self.db = db
        self.num_topics = len(self.top_words)
        self.model = model
        self.root = root
        self.presence_score = model.presence_score

        self.root = root

        self.button = None
        self.populate(self.presence_score)

    def template(self):
        self.parent = Canvas(self.root, width=1000, height=600, borderwidth=0, background="#D5DFF5")
        self.frame = Frame(self.parent, background="#D5DFF5")
        self.scroll = Scrollbar(self.root, orient="vertical", command=self.parent.yview)
        self.parent.configure(yscrollcommand=self.scroll.set)

        self.scroll.grid(row=1, column=1, sticky='nsew')  # pack(side="right", fill="y")
        self.parent.grid(row=1, column=0, sticky='nsew')  # pack(side="left", fill="both", expand=True)
        self.window = self.parent.create_window((80, 10), window=self.frame,
                                                anchor="nw", tags="self.frame")

        self.frame.bind("<Configure>", self.onFrameConfigure)

    def hover(self, color, event):
        canvas = event.widget
        canvas.configure(background=color)

    def populate(self, presence_score):
        for widget in self.root.winfo_children():
            widget.destroy()

        self.template()

        if self.button is not None:
            self.button.destroy()

        index_sorted = np.argsort(presence_score)[::-1]
        presence_score /= presence_score[index_sorted[0]]

        canvas = Canvas(self.frame, width=400, height=100, bd=0, bg='#D5DFF5')
        canvas.grid(row=0, column=0)
        canvas.create_text(200, 50, font=('', 16, 'bold'), text='Topics')

        row = 1
        for id in index_sorted:
            canvas_width = int(presence_score[id] * 800)
            if canvas_width < 80:
                canvas_width = 80
                canvas_height = 60
            elif canvas_width < 120:
                canvas_height = 60
            elif canvas_width < 180:
                canvas_height = 40
            else:
                canvas_height = 25

            canvas = Canvas(self.frame, width=canvas_width, height=canvas_height, bd=0, bg='#799FF2')
            canvas.grid(row=row, column=0, padx=5, pady=5)
            text = '{' + self.top_words[id][0] + ', ' + self.top_words[id][1] + ', ' + self.top_words[id][2] + '}'
            canvas.create_text(canvas_width/2, canvas_height/2, width=canvas_width, text=text)
            canvas.bind('<Enter>', functools.partial(self.hover, '#416DCC'))
            canvas.bind('<Leave>', functools.partial(self.hover, '#799FF2'))
            canvas.bind('<Button-1>', functools.partial(self.next_page, id, text))

            row += 1

    def onFrameConfigure(self, event):
        '''Reset the scroll region to encompass the inner frame'''
        self.parent.configure(scrollregion=self.parent.bbox("all"))

    def next_page(self, k, title, event):
        for widget in self.root.winfo_children():
            widget.destroy()

        self.template()

        if self.button is not None:
            self.button.destroy()

        #self.parent.itemconfig(self.window, x=10)
        words = self.top_words[k]
        canvas = Canvas(self.frame, width=400, height=100, bd=0, bg='#D5DFF5')
        canvas.grid(row=0, column=1, sticky='nesw')
        canvas.create_text(200, 50, font=('',14, 'bold'), text='Topic: '+title, width=400)

        canvas = Canvas(self.frame, width=200, height=30, bd=0, bg='#416DCC')
        canvas.grid(row=1, column=0, sticky='nesw')
        canvas.create_text(100, 15, width=200, text='words', font=('', 12, 'bold'))
        row = 2
        for term in words:
            canvas = Canvas(self.frame, width=200, height=25, bd=0, bg='#799FF2')
            canvas.grid(row=row, column=0, pady=0, padx=0, sticky='nesw')
            canvas.create_text(100, 25/2, width=200, text=term)
            #canvas.bind('<Enter>', functools.partial(self.hover, '#628AE3'))
            #canvas.bind('<Leave>', functools.partial(self.hover, '#799FF2'))
            row += 1


        canvas = Canvas(self.frame, width=400, height=30, bd=0, bg='#799FF2')
        canvas.grid(row=1, column=1, sticky='nesw')
        canvas.create_text(200, 15, width=400, text='related documents', font=('', 12, 'bold'))
        name_col = 'dist_topic'+str(k)
        num_docs = self.db.subset
        arr_theta_k = self.db.reader.select('theta', where='index<num_docs', columns=[name_col])[name_col].values
        index_sorted = np.argsort(arr_theta_k)[::-1]
        row = 2
        for i in range(20):
            id = index_sorted[i]
            canvas = Canvas(self.frame, width=400, height=25, bd=0, bg='white')
            canvas.grid(row=row, column=1, sticky='nesw')
            text = self.db.reader.select('corpus', where='index=id', columns=['title'])['title'].values[0]
            canvas.create_text(20, 13, width=400, text=text, anchor=W)
            canvas.bind('<Enter>', functools.partial(self.hover, '#B5CAF7'))
            canvas.bind('<Leave>', functools.partial(self.hover, 'white'))
            canvas.bind('<Button-1>', functools.partial(self.document_page, id))
            row += 1

        canvas = Canvas(self.frame, width=200, height=30, bd=0, bg='#416DCC')
        canvas.grid(row=1, column=2, sticky='nesw')
        canvas.create_text(100, 15, width=200, text='related topics', font=('', 12, 'bold'))
        beta = self.model.normalize()
        similar_topics = calc_similar(beta[k], beta, 1e-8)
        index_sorted = np.argsort(similar_topics)[::-1]
        row = 2
        for i in range(20):
            id = index_sorted[i]

            canvas = Canvas(self.frame, width=200, height=25, bd=0, bg='#799FF2')
            canvas.grid(row=row, column=2, padx=0, pady=0, sticky='nsew')
            text = '{' + self.top_words[id][0] + ', ' + self.top_words[id][1] + ', ' + self.top_words[id][2] + '}'
            canvas.create_text(100, 25/2, width=200, text=text)
            canvas.bind('<Enter>', functools.partial(self.hover, '#628AE3'))
            canvas.bind('<Leave>', functools.partial(self.hover, '#799FF2'))
            canvas.bind('<Button-1>', functools.partial(self.next_page, id, text))
            row += 1

        action = functools.partial(self.populate, self.presence_score)
        self.button = Button(self.root, text='Topic Page', command=action)
        self.button.grid(row=0, columns=1)#pack(side=BOTTOM)
        #self.button.lift(self.root)

    def document_page(self, id_doc, event):
        for widget in self.root.winfo_children():
            widget.destroy()

        self.template()

        if self.button is not None:        
            self.button.destroy()

        canvas = Canvas(self.frame, width=200, height=30, bd=0, bg='#416DCC')
        canvas.grid(row=1, column=0, sticky='nesw')
        canvas.create_text(100, 15, text='related topics', font=('', 12, 'bold'))

        frame_circle = Frame(self.frame, background='#799FF2')
        frame_circle.grid(row=2, column=0, sticky='nesw', rowspan=5)
        frame_circle.rowconfigure(0, weight=1)
        frame_circle.columnconfigure(0, weight=1)
        self.circle = Canvas(frame_circle, width=130, height=130, bg='white')
        self.circle.grid(row=0, column=0)

        theta_d = self.db.reader.select('theta', where='index=id_doc').values[0]
        index_sorted = np.argsort(theta_d)[::-1]
        length = len(np.where(theta_d*360 > 5)[0])
        if length > 15:
            length = 15
        index_sorted = index_sorted[:length]

        list_angle = list()
        list_item = list()
        start = 0
        for i in range(length):
            angle = theta_d[index_sorted[i]]*2*np.pi
            list_angle.append((frac(start), frac(start+angle)))
            #print list_angle[i]
            item = self.circle.create_arc((5,5,125,125), fill="#D5DFF5", outline="white",
                                          start=frac(start), extent=frac(angle))
            list_item.append(item)
            start += angle

        list_rec = list()
        row = 7
        for i in range(length):
            k = index_sorted[i]
            words = self.top_words[k]
            text = '{' + words[0] + ', ' + words[1] + ', ' + words[2] + '}'
            canvas = Canvas(self.frame, width=200, height=25, bd=0, bg='#799FF2')
            canvas.grid(row=row, column=0, sticky='nesw')
            canvas.create_text(100, 25/2, width=200, text=text)

            canvas.bind('<Enter>', functools.partial(self.hover_rec_to_pie, '#628AE3', '#779FF2', list_item[i]))
            canvas.bind('<Leave>', functools.partial(self.hover_rec_to_pie, '#799FF2', '#D5DFF5', list_item[i]))
            canvas.bind('<Button-1>', functools.partial(self.next_page, k, text))

            list_rec.append(canvas)

            row += 1
        origin_coord, R = (65,65), 60
        self.circle.bind('<Motion>', functools.partial(self.hover_pie_to_rec, origin_coord, R,
                                                       list_angle, list_item, list_rec))
        self.circle.bind('<Button-1>', functools.partial(self.click_pie, origin_coord, R, list_angle, index_sorted))


        document = self.db.reader.select('corpus', where='index=id_doc')
        title = document['title'].values[0]
        content = document['content'].values[0]
        canvas = Canvas(self.frame, width=400, height=100, bd=0, bg='#D5DFF5')
        canvas.grid(row=0, column=1, sticky='nesw')
        canvas.create_text(200, 50, font=('', 14, 'bold'), width=400, text=title)

        canvas = Canvas(self.frame, width=400, height=530, bd=0, bg='white')
        canvas.grid(row=1, column=1, rowspan=22, sticky='nsew')
        canvas.create_text(20, 20, anchor='nw', width=380, text=content)

        arr_similar = np.zeros(self.db.subset)
        i = 0
        while i < self.db.subset:
            #print i
            if (i+5000) >= self.db.subset:
                arr_index = list(range(i, self.db.subset))
            else:
                arr_index = list(range(i,i+5000))
            #print len(arr_index)
            chunk_theta = self.db.reader.select('theta', where='index=arr_index').values
            #print len(chunk_theta)
            arr_similar[arr_index] = calc_similar(theta_d, chunk_theta, 1e-8)
            i = i+5000
        index_sorted = np.argsort(arr_similar)[::-1]
        canvas = Canvas(self.frame, width=200, height=30, bd=0, bg='#416DCC')
        canvas.grid(row=1, column=2, sticky='nsew')
        canvas.create_text(100, 15, width=200, text='related document', font=('', 12, 'bold'))
        row = 2
        for i in range(20):
            id = index_sorted[i]
            title = self.db.reader.select('corpus', where='index=id', columns=['title'])['title'].values[0]
            canvas = Canvas(self.frame, width=200, height=25, bd=0, bg='#799FF2')
            canvas.grid(row=row, column=2, sticky='nsew')
            canvas.create_text(100, 25/2, width=200, text=title)
            canvas.bind('<Enter>', functools.partial(self.hover, '#416DCC'))
            canvas.bind('<Leave>', functools.partial(self.hover, '#799FF2'))
            canvas.bind('<Button-1>', functools.partial(self.document_page, id))
            row += 1

        action = functools.partial(self.populate, self.presence_score)
        self.button = Button(self.root, text='Topic Page', command=action)
        self.button.grid(row=0, columns=1)

    def hover_rec_to_pie(self, color_rec, color_pie, pie, event):
        canvas = event.widget
        self.circle.itemconfig(pie, fill=color_pie)
        canvas.configure(background=color_rec)

    def hover_pie_to_rec(self, origin_coord, R, list_angle, list_item, list_rec, event):
        for id in range(len(list_item)):
            self.circle.itemconfig(list_item[id], fill="#D5DFF5")
            list_rec[id].configure(background="#799FF2")
        x, y = event.x, event.y
        id = get_arc(x, y, origin_coord, R, list_angle)
        if id is not None:
            self.circle.itemconfig(list_item[id], fill="#799FF2")
            list_rec[id].configure(background="#628AE3")
        else:
            for id in range(len(list_item)):
                self.circle.itemconfig(list_item[id], fill="#D5DFF5")
                list_rec[id].configure(background="#799FF2")

    def click_pie(self, origin_coord, R, list_angle, index_sorted, event):
        x, y = event.x, event.y
        id = get_arc(x, y, origin_coord, R, list_angle)
        if id is not None:
            k = index_sorted[id]
            words = self.top_words[k]
            text = '{' + words[0] + ', ' + words[1] + ', ' + words[2] + '}'
            self.next_page(k, text, event)

def visualize(model, database, data, vocab_file):
    if not os.path.isfile(vocab_file):
        logging.error("File vocab %s doesn't exist")

    logging.info('Loading model...')
    if type(model) is str:
        if os.path.isfile(model):
            obj_model = LdaModel()
            obj_model.load(model)
        else:
            logging.error("File %s doesn't exist" %model)
    else:
        obj_model = model

    logging.info('Loading database...')
    if type(database) is str:
        if os.path.isfile(database):
            db = DataBase(database)
        else:
            logging.error("File %s doesn't exist" %database)
    else:
        db = database

    if type(data) is str:
        input_format = utilizies.check_input_format(data)
        if input_format == DataFormat.RAW_TEXT:
            db.store_from_raw_text(data)
        elif input_format == DataFormat.TERM_FREQUENCY:
            db.store_from_term_frequency(data, vocab_file)
        else:
            db.store_from_term_sequence(data, vocab_file)
    else:
        db.store_from_object(data)

    logging.info('Done!')

    root = Tk()
    root.columnconfigure(0, weight=1)
    root.rowconfigure(1, weight=1)
    TopicPage(root, obj_model, db, vocab_file)  # .grid(row=1)#pack(side="top", fill="both", expand=True)
    root.mainloop()
    db.reader.close()

if __name__ == '__main__':
    model = 'model-streaming-ope/model.h5'
    database = 'model-streaming-ope/db.h5'
    vocab_file = os.path.expanduser('~/tmlib_data/ap_train_raw/vocab.txt') #WikiStream/current_vocab.txt'
    data = 'examples/ap/data/ap_train_raw.txt' #'/home/khangtg/Desktop/topicmodel-lib/examples/ap/data/ap_train_raw.txt'
    visualize(model, database, data, vocab_file)







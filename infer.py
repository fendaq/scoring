from get_model import *
import pdf2image
from pyson.utils import *

# get graph
inputs = tf.placeholder(tf.float32, [None, None, None, 1], name='inputs')
with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
    model = get_model('dots', inputs)
sess = tf.Session()
saver = tf.train.Saver()
saver.restore(sess, tf.train.latest_checkpoint('checkpoint/'))



def convert(path):
    pdf = pdf2image.convert_from_path(path)
    return np.array(pdf.pop())

def get_abcd_pads(x_half, mask):
    def predict(pad):
        x = np.split(cv2.resize(pad, (4*(pad.shape[1]//4), pad.shape[0])), 4, axis=1)
        x = [np.mean(_) for _ in x]
        p = np.argmin(x)
        return chr(ord('a')+p)
    def putText(img, pos, text):
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img,text,pos, font, 1,(0,0,255),2,cv2.LINE_AA)
        return img
    
    h,w = x_half.shape[:2]
    mask = cv2.resize(mask, (w, h))
    y = x_half
    x = mask
    
    print(x.shape, y.shape)
    x = cv2.threshold(x, 0, 255, cv2.THRESH_OTSU+cv2.THRESH_BINARY_INV)[1]
    x = cv2.morphologyEx(x, cv2.MORPH_CLOSE, np.ones([1, 20]))
    x = cv2.morphologyEx(x, cv2.MORPH_OPEN, np.ones([1, 100]))
#     show(x)
    cnts = findContours(x)[0]
    pads = []
    def score(cnt):
        x,y,w,h = cv2.boundingRect(cnt)
        return x*10+y
    results = {}
    cnts = sorted(cnts, key=score)
    print(y.shape)
    for i, cnt in enumerate(cnts):
        x_,y_,w_,h_ = cv2.boundingRect(cnt)
        pad = y[y_:y_+h_, x_:x_+w_]
        pads.append(pad)
        # prediction value on current pad
        p_val = predict(pad)
        x = putText(y, (x_-30, y_+10), '{}'.format(p_val))
        # save to result dict
        results[i+1] = p_val
    return results


def process_pdf(pdf):
    print(pdf)
    img = convert(pdf)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = img[:img.shape[0]//2]/127.5-1
    
    img = cv2.resize(img, (1024, 512))
    img = np.expand_dims(img, 0)
    img = np.expand_dims(img, -1)
    
    img_out = sess.run(model['outputs'][0,...,0], {model['inputs']:img})*255
    img_out = img_out.astype('uint8')
    img_in = convert(pdf)
    img_half = img_in[:img_in.shape[0]//2]
    results = get_abcd_pads(img_half, img_out)
    with open(pdf.replace('.pdf','.json'), 'w') as f:
        json.dump(results, f)
    print('INFO: dump at:', pdf.replace('.pdf', '.json'))
    print(len(results))
    print('-'*100)
if __name__ == '__main__':
    for pdf_path in sorted(glob('data/*.pdf')):
        process_pdf(pdf_path)



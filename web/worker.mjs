import {createHandler} from "./runtime.mjs";

const handle = createHandler();
self.onmessage = ({data}) => handle(data, message => self.postMessage(message));

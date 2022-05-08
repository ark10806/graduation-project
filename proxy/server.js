var express = require('express')
var bodyParser = require('body-parser')
var http = require('http')
var app = express()
const getMeme = require('./service')

const cors = require('cors')
const PORT = 5000
var httpServer = http.createServer(app)

const file_limit = "300mb"
app.use(bodyParser.json({ limit: file_limit }));
app.use(bodyParser.urlencoded({limit: file_limit, extended: true}));
app.use(
	cors({
		origin: true,
        credentials: true,
	}),
);

app.post('/',  function(req, res){
	console.log('est');
	var dic = {
		'result': 'hello'
	}
	res.write(JSON.stringify(dic))
	res.end()
})

app.get('/hello', async function(req, res){
	try{
        result = await getMeme('cat')
        let json = JSON.stringify(result)
        res.write(json)
        res.end()
    } catch (error){
        console.log(`[Error]: GET meme with ${error}`)
    }
})

app.get('/meme', async function(req, res){
    try{
        const msg = req.query.message
        console.log(msg)
        result = await getMeme(msg)
        let json = JSON.stringify(result)
        res.write(json)
        res.end()
    } catch (error){
        console.log(`[Error]: GET meme with ${error}`)
    }
})


httpServer.listen(PORT, function(){
	console.log(`listening at ${PORT}`);
});

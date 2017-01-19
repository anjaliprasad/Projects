package project.door.lock.pack;

import android.app.Activity;
import android.bluetooth.BluetoothAdapter;
import android.bluetooth.BluetoothDevice;
import android.content.Intent;
import android.content.SharedPreferences;
import android.content.SharedPreferences.Editor;
import android.os.Bundle;
import android.os.Handler;
import android.os.Message;
import android.preference.PreferenceManager;
import android.util.Log;
import android.view.KeyEvent;
import android.view.Menu;
import android.view.MenuInflater;
import android.view.MenuItem;
import android.view.View;
import android.view.Window;
import android.view.View.OnClickListener;
import android.view.inputmethod.EditorInfo;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ListView;
import android.widget.TextView;
import android.widget.Toast;

public class MainActivity extends Activity implements OnClickListener
{
	private static final String TAG = "Bluetooth Remote";
    private static final boolean D = true;

    
    public static final int MESSAGE_STATE_CHANGE = 1;
    public static final int MESSAGE_READ = 2;
    public static final int MESSAGE_WRITE = 3;
    public static final int MESSAGE_DEVICE_NAME = 4;
    public static final int MESSAGE_TOAST = 5;

    
    public static final String DEVICE_NAME = "device_name";
    public static final String TOAST = "toast";

   
    private static final int REQUEST_CONNECT_DEVICE = 1;
    private static final int REQUEST_ENABLE_BT = 2;

    
    private TextView mTitle;
    private ListView mConversationView;
    private EditText mOutEditText;
    private Button mSendButton;

    
    private String mConnectedDeviceName = null;
    
    private StringBuffer mOutStringBuffer;
    
    private BluetoothAdapter mBluetoothAdapter = null;
   
    private static BluetoothChatService mChatService = null;
    
    public ListView apps_list;
    
    EditText txt_user,txt_pass;
    Button btn_submit;
    boolean connected_to_device=false;
    String serial;

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        // Set up the window layout
        requestWindowFeature(Window.FEATURE_CUSTOM_TITLE);
        setContentView(R.layout.main);
        txt_user=(EditText)findViewById(R.id.main_txt_user);
        txt_pass=(EditText)findViewById(R.id.main_txt_pass);
        btn_submit=(Button)findViewById(R.id.main_btn_submit);
        btn_submit.setOnClickListener(this);
       
        getWindow().setFeatureInt(Window.FEATURE_CUSTOM_TITLE, R.layout.custom_title);

        // Set up the custom title
        mTitle = (TextView) findViewById(R.id.title_left_text);
        mTitle.setText(R.string.app_name);
        mTitle = (TextView) findViewById(R.id.title_right_text);

        // Get local Bluetooth adapter
        mBluetoothAdapter = BluetoothAdapter.getDefaultAdapter();

        // If the adapter is null, then Bluetooth is not supported
        if (mBluetoothAdapter == null) {
            Toast.makeText(this, "Bluetooth is not available", Toast.LENGTH_LONG).show();
            finish();
            return;
        }
    }

    @Override
    public void onStart() {
        super.onStart();
        
        if (!mBluetoothAdapter.isEnabled()) 
        {
            Intent enableIntent = new Intent(BluetoothAdapter.ACTION_REQUEST_ENABLE);
            startActivityForResult(enableIntent, REQUEST_ENABLE_BT);
       
        } else {
            if (mChatService == null) setupChat();
        }
    }

    @Override
    public synchronized void onResume() {
        super.onResume();
       
        if (mChatService != null) {
            // Only if the state is STATE_NONE, do we know that we haven't started already
            if (mChatService.getState() == BluetoothChatService.STATE_NONE) {
              // Start the Bluetooth chat services
              mChatService.start();
            }
        }
    }

    private void setupChat() 
    {
        
        mChatService = new BluetoothChatService(this, mHandler);        
        mOutStringBuffer = new StringBuffer("");
    }
    
    public static BluetoothChatService getBluetoothChatObject()
    {
    	return  mChatService;
    }

    @Override
    public synchronized void onPause() {
        super.onPause();
      
    }

    @Override
    public void onStop() {
        super.onStop();
        
    }

    @Override
    public void onDestroy() {
        super.onDestroy();
        // Stop the Bluetooth chat services
        if (mChatService != null) mChatService.stop();
        
    }

    private void ensureDiscoverable() {
       
        if (mBluetoothAdapter.getScanMode() !=
            BluetoothAdapter.SCAN_MODE_CONNECTABLE_DISCOVERABLE) {
            Intent discoverableIntent = new Intent(BluetoothAdapter.ACTION_REQUEST_DISCOVERABLE);
            discoverableIntent.putExtra(BluetoothAdapter.EXTRA_DISCOVERABLE_DURATION, 300);
            startActivity(discoverableIntent);
        }
    }

    /**
     * Sends a message.
     * @param message  A string of text to send.
     */
    private void sendMessage(String message) {
        // Check that we're actually connected before trying anything
        if (mChatService.getState() != BluetoothChatService.STATE_CONNECTED) {
            Toast.makeText(this, R.string.not_connected, Toast.LENGTH_SHORT).show();
            return;
        }

        // Check that there's actually something to send
        if (message.length() > 0) {
            // Get the message bytes and tell the BluetoothChatService to write
            byte[] send = message.getBytes();
            mChatService.write(send);

            // Reset out string buffer to zero and clear the edit text field
            mOutStringBuffer.setLength(0);
            mOutEditText.setText(mOutStringBuffer);
        }
    }

    // The action listener for the EditText widget, to listen for the return key
    private TextView.OnEditorActionListener mWriteListener =
        new TextView.OnEditorActionListener() {
        public boolean onEditorAction(TextView view, int actionId, KeyEvent event) {
            // If the action is a key-up event on the return key, send the message
            if (actionId == EditorInfo.IME_NULL && event.getAction() == KeyEvent.ACTION_UP) {
                String message = view.getText().toString();
                sendMessage(message);
            }
            if(D) Log.i(TAG, "END onEditorAction");
            return true;
        }
    };

    // The Handler that gets information back from the BluetoothChatService
    private final Handler mHandler = new Handler() {
        @Override
        public void handleMessage(Message msg) {
            switch (msg.what) {
            case MESSAGE_STATE_CHANGE:
                if(D) Log.i(TAG, "MESSAGE_STATE_CHANGE: " + msg.arg1);
                switch (msg.arg1) 
                {
                case BluetoothChatService.STATE_CONNECTED:
                    mTitle.setText(R.string.title_connected_to);
                    mTitle.append(mConnectedDeviceName);
                    //Intent it=new Intent(MainActivity.this,MenuActivity.class);
                    //startActivity(it);
                    connected_to_device=true;
                    char[] c=serial.toCharArray();    
                    byte[] b=new byte[c.length];
                    for(int i=0;i<c.length;i++)
                    {
                      b[i]=(byte)c[i];
                    }
                    mChatService.write(b);
                    break;
                    
                case BluetoothChatService.STATE_CONNECTING:
                    mTitle.setText(R.string.title_connecting);
                    break;
                    
                case BluetoothChatService.STATE_LISTEN:
                case BluetoothChatService.STATE_NONE:
                    mTitle.setText(R.string.title_not_connected);
                    break;
                }
                break;
            
            case MESSAGE_DEVICE_NAME:
                // save the connected device's name
                mConnectedDeviceName = msg.getData().getString(DEVICE_NAME);
                Toast.makeText(getApplicationContext(), "Connected to "
                               + mConnectedDeviceName, Toast.LENGTH_SHORT).show();
                break;
            case MESSAGE_TOAST:
                Toast.makeText(getApplicationContext(), msg.getData().getString(TOAST),
                               Toast.LENGTH_SHORT).show();
                break;
                
              case MESSAGE_READ:
            	
            	String data=new String((byte[])msg.obj).trim();
            	if(data.contains("X"))
            	{
            		 Toast.makeText(getApplicationContext(),"Wrong Username or Password",
                             Toast.LENGTH_SHORT).show();
            	}
            	else if(data.contains("K"))
            	{
            		 Intent it=new Intent(MainActivity.this,MenuActivity.class);
                     startActivity(it);
            	}
    
            }
        }
    };

    public void onActivityResult(int requestCode, int resultCode, Intent data) {
        if(D) Log.d(TAG, "onActivityResult " + resultCode);
        switch (requestCode) {
        case REQUEST_CONNECT_DEVICE:
            // When DeviceListActivity returns with a device to connect
            if (resultCode == Activity.RESULT_OK) {
                // Get the device MAC address
                String address = data.getExtras()
                                     .getString(DeviceListActivity.EXTRA_DEVICE_ADDRESS);
                // Get the BLuetoothDevice object
                BluetoothDevice device = mBluetoothAdapter.getRemoteDevice(address);
                // Attempt to connect to the device
                mChatService.connect(device);
            }
            break;
        case REQUEST_ENABLE_BT:
            // When the request to enable Bluetooth returns
            if (resultCode == Activity.RESULT_OK) {
                
                setupChat();
            } else {
                
                Log.d(TAG, "BT not enabled");
                Toast.makeText(this, R.string.bt_not_enabled_leaving, Toast.LENGTH_SHORT).show();
                finish();
            }
        }
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        MenuInflater inflater = getMenuInflater();
        inflater.inflate(R.menu.option_menu, menu);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        switch (item.getItemId()) {
        case R.id.register:
            // Launch the DeviceListActivity to see devices and do scan
        	SharedPreferences pref=PreferenceManager.getDefaultSharedPreferences(getBaseContext());
			String code=pref.getString("vcode",null);
			if(code!=null)
			{
				 Intent serverIntent = new Intent(this,VerifyActivity.class);
				 Bundle b=new Bundle();
				 b.putInt("which",2);
				 serverIntent.putExtra("data",b);
		          startActivity(serverIntent);
			}
			else
			{
			 Intent serverIntent = new Intent(this,RegisterActivity.class);
			 Bundle b=new Bundle();
			 b.putInt("which",1);
			 serverIntent.putExtra("data",b);
	          startActivity(serverIntent);
			}
            return true;
      
        }
        return false;
    }

	public void onClick(View v) {
		// TODO Auto-generated method stub
		String user=txt_user.getText().toString();
		String pass=txt_pass.getText().toString();
		
		SharedPreferences prefs = PreferenceManager.getDefaultSharedPreferences(getBaseContext());
		String duser=prefs.getString("user",null);
		String dpass=prefs.getString("pass",null);
		serial=prefs.getString("serial",null);
		serial=serial+"\n";
		
		
		if(duser!=null&&dpass!=null)
		{
		if(duser.equals(user)&&dpass.equals(pass))
		{
			  if(!connected_to_device)
			  {
			  Intent serverIntent = new Intent(this, DeviceListActivity.class);
	          startActivityForResult(serverIntent, REQUEST_CONNECT_DEVICE);
			  }
			  else
			  {
				  mChatService.write((serial+"\n").getBytes());
			  }
        }
		else
		{
			Toast.makeText(this,"Login Failed",Toast.LENGTH_LONG).show();
		}
		}
		else
		{
			
			 Intent serverIntent = new Intent(this,RegisterActivity.class);			 
	          startActivity(serverIntent);
			
		}
		
		
		
	}


		
	

}